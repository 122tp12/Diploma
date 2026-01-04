import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import numpy as np


class EarlyStopper:
    def __init__(self, patience=150, min_delta=0, path='tmp_best_current_model.pt'):
        """
        patience: скільки епох чекати після останнього покращення.
        min_delta: мінімальна зміна, яка вважається покращенням (щоб не реагувати на шум 0.000001).
        path: куди зберігати найкращу модель.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss, model):
        # Якщо loss покращився (став меншим за поточний найкращий)
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
            model.save_model(self.path)
        # Якщо loss не покращився
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    def load_best_model(self, model, device):
        model.load_model(self.path, device)
        return model
                
class EGAT_model(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, heads=4):
        super().__init__()
        
        self.conv1 = GATv2Conv(
            in_channels=in_channels, 
            out_channels=hidden_channels, 
            heads=heads, 
            edge_dim=edge_dim,
            concat=True
        )
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels * heads)
        self.conv2 = GATv2Conv(
            in_channels=hidden_channels * heads, 
            out_channels=out_channels, 
            heads=1, 
            edge_dim=edge_dim,
            concat=False
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=5e-4)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        
        return x
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path, device):
        checkpoint = torch.load(path, map_location=device)
        if isinstance(checkpoint, dict):
            self.load_state_dict(checkpoint)
        elif hasattr(checkpoint, 'state_dict'):
            self.load_state_dict(checkpoint.state_dict())
        else:
            raise RuntimeError(f'Unrecognized checkpoint format: {type(checkpoint)}')
        self.eval()
        return self

def accuracy(pred_y, y):
    return ((pred_y == y).sum() / len(y)).item()
def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index, data.edge_attr)
    acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    return acc

def train(model, data):
    device = next(model.parameters()).device
    weight = torch.tensor([1.0, 5.0], device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    optimizer = model.optimizer
    epochs = 2000
    
    early_stopper = EarlyStopper(patience=300, path='./checkpoints/tmp_best_current_model.pt')

    model.train()
    for epoch in range(epochs+1):
        # Training
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Validation
        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
        val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])

        # Print metrics every 10 epochs
        if(epoch % 10 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.5f} | Train Acc: '
                  f'{acc*100:>6.2f}% | Val Loss: {val_loss:.5f} | '
                  f'Val Acc: {val_acc*100:.2f}%')
            
        early_stopper(val_loss, model)
            
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch}!")
            model=early_stopper.load_best_model(model, device)
            break
          
    return model
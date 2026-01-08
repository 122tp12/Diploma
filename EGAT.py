from sched import scheduler
from tabnanny import verbose
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
        current_loss = val_loss.item() if torch.is_tensor(val_loss) else val_loss

        if current_loss < (self.best_loss - self.min_delta):
            self.best_loss = current_loss
            self.counter = 0
            model.save_model(self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    def load_best_model(self, model, device):
        model.load_model(self.path, device)
        return model
                
class EGAT_model(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, num_hiden_layers=1, heads=4):
        super().__init__()
        
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        # 1. Input Layer
        # 'in_channels' -> 'hidden_channels * heads'
        self.convs.append(GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            edge_dim=edge_dim,
            concat=True
        ))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads))

        # 2. Hidden Layers
        # hidden * heads
        for _ in range(num_hiden_layers):
            self.convs.append(GATv2Conv(
                in_channels=hidden_channels * heads, # Previous layer output
                out_channels=hidden_channels,        # Hidden out (concat=True -> *heads)
                heads=heads,
                edge_dim=edge_dim,
                concat=True
            ))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads))

        # 3. Output Layer
        # 'hidden_channels * heads' -> 'out_channels'
        # Тут concat=False, тому heads=1 (або усереднюємо, якщо heads>1)
        self.convs.append(GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,
            edge_dim=edge_dim,
            concat=False
        ))

    def forward(self, x, edge_index, edge_attr):
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr)
            x = self.bns[i](x)      # Нормалізація
            x = F.elu(x)
            x = F.dropout(x, p=0.6, training=self.training)

        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)
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
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr)
        acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    return acc

def train_step(model, data, criterion, optimizer):
    model.train() # Вмикаємо режим навчання (важливо для Dropout)
    
    # 1. Forward
    out = model(data.x, data.edge_index, data.edge_attr)
    
    # 2. Loss
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    
    # 3. Backward
    optimizer.zero_grad() # Очищаємо старі градієнти
    loss.backward()       # Рахуємо нові
    optimizer.step()      # Оновлюємо ваги
    
    acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
    
    return loss.item(), acc

@torch.no_grad()
def validate_step(model, data, criterion):
    model.eval()
    
    out = model(data.x, data.edge_index, data.edge_attr)
    val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
    val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])
    
    return val_loss.item(), val_acc

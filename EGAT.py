import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.nn import BatchNorm1d, Linear
import numpy as np



class EarlyStopper:
    def __init__(self, patience=150, min_delta=0, path='tmp_best_current_model.pt'):
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

class EGATLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, edge_dim, heads=4, concat=True):
        super().__init__()
        
        self.concat = concat
        self.out_channels = out_channels
        self.total_out_dim = out_channels * heads if concat else out_channels
        
        # 1. Node update
        self.conv = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            edge_dim=edge_dim,
            concat=concat,
            add_self_loops=False
        )
        
        # 2. Edge Update
        self.lin_node = Linear(3 * self.total_out_dim, edge_dim)
        self.lin_edge = Linear(edge_dim, edge_dim)
        self.lin_reduce = Linear(2 * edge_dim, edge_dim)

    def forward(self, x, edge_index, edge_attr):
        # 1: h'
        x_new = self.conv(x, edge_index, edge_attr=edge_attr)
        
        # 2: f'
        row, col = edge_index
        h_i = x_new[row]
        h_j = x_new[col]
        
        feat_cat = torch.cat([h_i, h_j, torch.abs(h_i - h_j)], dim=-1)
        r_ij = F.elu(self.lin_node(feat_cat))
        
        t_ij = F.elu(self.lin_edge(edge_attr))

        edge_attr_new = F.elu(self.lin_reduce(torch.cat([r_ij, t_ij], dim=-1)))
        
        return x_new, edge_attr_new

class EGAT_model(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, num_hidden_layers=1, heads=4, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.layers = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.edge_bns = torch.nn.ModuleList()
        
        # --- 1. Input layer ---
        self.layers.append(EGATLayer(
            in_channels=in_channels,
            out_channels=hidden_channels,
            edge_dim=edge_dim,
            heads=heads,
            concat=True
        ))
        self.bns.append(BatchNorm1d(hidden_channels * heads))
        self.edge_bns.append(BatchNorm1d(edge_dim))

        # --- 2. Hidden layers ---
        for _ in range(num_hidden_layers):
            self.layers.append(EGATLayer(
                in_channels=hidden_channels * heads,
                out_channels=hidden_channels,
                edge_dim=edge_dim,
                heads=heads,
                concat=True
            ))
            self.bns.append(BatchNorm1d(hidden_channels * heads))
            self.edge_bns.append(BatchNorm1d(edge_dim))

        # --- 3. Output layer ---
        self.final_conv = GATv2Conv(
            in_channels=hidden_channels * heads,
            out_channels=out_channels,
            heads=1,
            edge_dim=edge_dim,
            concat=False
        )
    
    def forward(self, x, edge_index, edge_attr):
        
        for i, layer in enumerate(self.layers):
            # 1. Residual Connection
            x_in = x
            edge_attr_in = edge_attr

            # 2.
            x, edge_attr = layer(x, edge_index, edge_attr)
            
            # 3. Node (Residual + Norm + Act + Dropout)
            if i > 0 and x.shape == x_in.shape:
                x = x + x_in
            
            x = self.bns[i](x)      # BatchNorm
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training) 

            # 4. Edge (Residual + Norm + Act + Dropout)
            if i > 0 and edge_attr.shape == edge_attr_in.shape:
                edge_attr = edge_attr + edge_attr_in

            edge_attr = self.edge_bns[i](edge_attr) # BatchNorm
            edge_attr = F.elu(edge_attr)
            edge_attr = F.dropout(edge_attr, p=self.dropout, training=self.training)

        # Final Layer
        x = self.final_conv(x, edge_index, edge_attr=edge_attr)
        
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
    model.train()
    
    # 1. Forward
    out = model(data.x, data.edge_index, data.edge_attr)
    
    # 2. Loss
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    
    # 3. Backward
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    
    acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
    
    return loss.item(), acc

@torch.no_grad()
def validate_step(model, data, criterion):
    model.eval()
    
    out = model(data.x, data.edge_index, data.edge_attr)
    val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
    val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])
    
    return val_loss.item(), val_acc

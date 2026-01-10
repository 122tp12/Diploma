import csv
from genericpath import isfile
import os
import json
from posixpath import join
import time
import matplotlib.pyplot as plt
from typing import List
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import datetime

import torch
import gc

from multiprocessing import Pool

from EGAT import EGAT_model, EarlyStopper, train_step, validate_step
import features

# To save dependencies:
# pip freeze > requirements.txt

def plot_strokes(strokes: dict, clasified):
    j=0
    for j in range(strokes.__len__()):
        points=strokes['t'+j.__str__()]
        current_x = points[0][0]
        current_y = points[0][1]
        current_t=points[0][2]
        current_f=points[0][3]
        stroke_points = [(-current_x, current_y, current_t, current_f)]
        if len(points)<2:
            continue
        i=2
        vx, vy, vt, vf = points[1][0], points[1][1], points[1][2], points[1][3]

        current_x += vx
        current_y += vy
        current_t += vt
        current_f += vf
        stroke_points.append((-current_x, current_y, current_t, current_f))

        while i < len(points):
            vx += points[i][0]
            vy+=points[i][1]
            vt+=points[i][2]
            vf+=points[i][3]
            
            current_x += vx
            current_y += vy
            current_t += vt
            current_f += vf
            stroke_points.append((-current_x, current_y, current_t, current_f))
            
            i+=1

        xs, ys, ts, fs = zip(*stroke_points)
        if type(clasified)==dict:
            if clasified['t'+j.__str__()]==1:
                plt.plot(xs, ys, color='blue', linewidth=1)
            elif clasified['t'+j.__str__()]==-1:
                plt.plot(xs, ys, color='green', linewidth=5)
            else:
                plt.plot(xs, ys, color='red', linewidth=1)
        else:
            if clasified[j]==1:
                plt.plot(xs, ys, color='blue', linewidth=1)
            elif clasified[j]==-1:
                plt.plot(xs, ys, color='green', linewidth=5)
            else:
                plt.plot(xs, ys, color='red', linewidth=1)
            
        
        j+=1

    plt.axis('equal')
    plt.gca().axis('off')
    plt.title("Render InkML")
    plt.show()

def _load_batch(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt['strokes'], ckpt['labels']
def _save_data_batch(data: Data, path: str):
    torch.save(data.to(torch.device('cpu')), path)

def load_data_batch(path: str, device) -> Data:
    data = torch.load(path, map_location=device, weights_only=False)
    return data

def process_data_batch(batch: str, device, proxy_threshold:float, time_threshold:float) -> Data:
    if os.path.exists(batch[:-3]+"_proc.pt"):
        return load_data_batch(batch[:-3]+"_proc.pt", device=device)

    graphs, true_y_graphs=_load_batch(batch, device)

    x_list = []
    edge_index_list = []
    edge_attr_list = []
    y_list = []

    offset=0
    
    for i in range(graphs.__len__()):
        strokes=graphs[i]
        current_y=true_y_graphs[i]

        dict_features=features.extract_stroke_features(strokes, offset, proxy_threshold, time_threshold)

        node_vals = list(dict_features["nodes"].values())
        x_tmp = torch.tensor(list(zip(*node_vals)), dtype=torch.float)
        x_list.append(x_tmp)

        edge_index_tmp = torch.tensor(dict_features["edge_index"], dtype=torch.long).t().contiguous()
        edge_index_list.append(edge_index_tmp)

        edge_vals = list(dict_features["edges_features"].values())
        edge_attr_tmp = torch.tensor(list(zip(*edge_vals)), dtype=torch.float)
        edge_attr_list.append(edge_attr_tmp)

        y_tmp = torch.tensor(current_y, dtype=torch.long)
        y_list.append(y_tmp)

        offset += x_tmp.size(0)

    x = torch.cat(x_list, dim=0)
    edge_attr = torch.cat(edge_attr_list, dim=0)
    true_y = torch.cat(y_list, dim=0)
    edge_index = torch.cat(edge_index_list, dim=1)

    data=Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=true_y,
        )
    
    if data.x is None or data.edge_index is None or data.edge_attr is None:
        raise ValueError("Data object must have 'x', 'edge_index', and 'edge_attr' attributes.")
    
    scaler = StandardScaler()
    data.x = torch.from_numpy(scaler.fit_transform(data.x)).float()
    data.edge_attr = torch.from_numpy(scaler.fit_transform(data.edge_attr)).float()

    data.train_mask, data.val_mask = features.get_masks(true_y.__len__())


    data.x = data.x.to(device)
    data.edge_attr = data.edge_attr.to(device)
    data.edge_index = data.edge_index.to(device)
    data.y = data.y.to(device)
    data.train_mask = data.train_mask.to(device)
    data.val_mask = data.val_mask.to(device)

    _save_data_batch(data, batch[:-3]+"_proc.pt")
    return data

def save_config(config: dict, path: str):
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)
def read_config(path: str):
    with open(path, 'r') as f:
        return json.load(f)

def main_train_loop(device)-> tuple[List[int], EGAT_model]:
    trs=read_config("./batches/setings.json")
    configs = {
        "out_channels": 2,
        "hidden_channels": 10,
        "hidden_layers": 2,
        "heads": 4,
        "lr": 0.05,
        "weight_decay": 5e-4,
        "batch_size": 16, # Change manualy
        "edge_treshhold": 80, # Change manualy
        "epochs": 1000,
        "factor": 0.5,
        "early_stopper_patience": 150,
        "scheduler_patience": 10,
        "scheduler_threshold": 0.0001,

        "proxy_threshold" : trs["proxy_threshold"],
        "time_threshold" : trs["time_threshold"],
        "features": trs["features"], 
        
        "description": "EGAT model training run"
    }
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f'./checkpoints/run_{timestamp}'
    os.makedirs(run_dir, exist_ok=True)
    
    save_config(configs, join(run_dir, 'config.json'))

    path = './batches/'
    files = [f for f in os.listdir(path) if f.endswith('_proc.pt')]

    temp_data = load_data_batch(join(path, files[0]), device)

    model = EGAT_model(
        in_channels=temp_data.x.size(1),
        hidden_channels=configs["hidden_channels"],
        out_channels=configs["out_channels"],
        edge_dim=temp_data.edge_attr.size(1),
        heads=configs["heads"],
        num_hiden_layers=configs["hidden_layers"]
    ).to(device)
    
    del temp_data 
    torch.cuda.empty_cache()

    optimizer = torch.optim.Adam(model.parameters(), lr=configs["lr"], weight_decay=configs["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
            mode='min', 
            factor=configs["factor"],
            patience=configs["scheduler_patience"],
            threshold=configs["scheduler_threshold"]
            )
    
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0], device=device))

    log_file = join(run_dir, 'metrics.csv')
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr', 'time'])

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    early_stopper = EarlyStopper(patience=configs["early_stopper_patience"], path='./checkpoints/tmp_best_current_model.pt')

    for epoch in range(configs["epochs"]):
        start_time = time.time()

        epoch_train_loss = 0
        epoch_val_loss = 0
        epoch_train_acc = 0
        epoch_val_acc = 0
        count = 0
        
        gc.collect()
        torch.cuda.empty_cache()
        
        for file_name in files:
            
            
            full_path = join(path, file_name)
            data = load_data_batch(full_path, device)

            loss, acc = train_step(model, data, criterion, optimizer)
            val_loss, val_acc = validate_step(model, data, criterion)
            
            epoch_train_loss += loss
            epoch_train_acc += acc
            
            epoch_val_loss += val_loss
            epoch_val_acc += val_acc
            count += 1
            
            del data
            gc.collect()
            torch.cuda.empty_cache()
        
        avg_train_loss = epoch_train_loss / count
        avg_train_acc = epoch_train_acc / count
        avg_val_loss = epoch_val_loss / count
        avg_val_acc = epoch_val_acc / count

        scheduler.step(avg_val_loss)
        early_stopper(avg_val_loss, model)

        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - start_time
        
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc, current_lr, epoch_time])
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)

        if epoch % 1 == 0:
            print(f'Epoch {epoch:>3} | Train Loss: {avg_train_loss:.5f} | Train Acc: '
                  f'{avg_train_acc*100:>6.2f}% | Val Loss: {avg_val_loss:.5f} | '
                  f'Val Acc: {avg_val_acc*100:.2f}% | LR: {current_lr:.6f}')
            
        if epoch+1 % 500==0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'criterion_dict': criterion.state_dict(),
                'loss': avg_val_loss,
                'config': configs
            }
            torch.save(checkpoint, join(run_dir, 'last_checkpoint.pt'))

        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch}!")
            model=early_stopper.load_best_model(model, device)
            
            break

    
    model.save_model(join(run_dir, 'final_model.pt'))

    return model

def _config_get_feature_names() -> dict:
    dummy_stroke = [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
    dummy_strokes = [dummy_stroke, dummy_stroke]

    feature = features.extract_stroke_features(dummy_strokes, 0, 1000.0, 1000.0)

    # Extract the keys directly from the result
    node_keys = list(feature["nodes"].keys())
    edge_keys = list(feature["edges_features"].keys())

    return {
        "node_features": node_keys,
        "edge_features": edge_keys,
        "num_node_features": len(node_keys),
        "num_edge_features": len(edge_keys)
    }

if __name__ == "__main__":
    configs={
        "proxy_threshold":80.0,
        "time_threshold":2.0,
        "features": _config_get_feature_names()
    }
    save_config(configs, "./batches/setings.json")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = './batches/'
    
    all_files = [f for f in os.listdir(path) if (isfile(join(path, f)) and f.endswith('.pt') and not f.endswith('_proc.pt'))]
    files_to_process = [f for f in all_files if not os.path.exists(join(path, f[:-3]+"_proc.pt"))]

    tasks = [(join(path, f), device, configs['proxy_threshold'], configs['time_threshold']) for f in files_to_process]

    with Pool(processes=10, maxtasksperchild=1) as pool:
        pool.starmap(process_data_batch, tasks)
    
    print("All processes finished.")
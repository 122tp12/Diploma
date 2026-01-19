import csv
from genericpath import isfile
import os
import json
from pdb import run
from posixpath import join
import random
import time
import matplotlib.pyplot as plt
from typing import List
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import StandardScaler
import datetime

from torch_geometric.loader import DataLoader

import torch
import gc

from multiprocessing import Pool

from EGAT import EGAT_model, EarlyStopper, test, train_step, validate_step
import StrokeGraphDataset_class
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

def process_data_batch(batch: str, destination: str, device, proxy_threshold:float, time_threshold:float) -> Data:
    if os.path.exists(destination):
        return load_data_batch(destination, device=device)

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
    
    data.edge_index, data.edge_attr = to_undirected(
        data.edge_index, 
        data.edge_attr, 
        reduce='mean' 
    )

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

    _save_data_batch(data, destination)
    return data

def save_config(config: dict, path: str):
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)
def read_config(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def set_batch_masks(data: Data, mode: str):
    """
    Overwrites masks so the model sees the WHOLE graph as either Train, Val, or Test.
    """
    num_nodes = data.x.size(0)
    if mode == 'train':
        data.train_mask = torch.ones(num_nodes, dtype=torch.bool, device=data.x.device)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
    elif mode == 'val':
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
        data.val_mask = torch.ones(num_nodes, dtype=torch.bool, device=data.x.device)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
    elif mode == 'test':
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=data.x.device)
        data.test_mask = torch.ones(num_nodes, dtype=torch.bool, device=data.x.device)
    return data

def main_train_loop(device, setting, dir_of_batches:str)-> tuple[List[int], EGAT_model]:
    STOP_FILE = "stop_training.txt"
    if os.path.exists(STOP_FILE):
        os.remove(STOP_FILE)

    trs=read_config(join(dir_of_batches, "setings.json"))
    configs = {
        "out_channels": setting["out_channels"],
        "hidden_channels": setting["hidden_channels"],
        "hidden_layers": setting["hidden_layers"],
        "heads": setting["heads"],
        "lr": setting["lr"],
        "weight_decay": setting["weight_decay"],
        "batch_size": setting["batch_size"],
        "epochs": setting["epochs"],
        "factor": setting["factor"],
        "early_stopper_patience": setting["early_stopper_patience"],
        "scheduler_patience": setting["scheduler_patience"],
        "scheduler_threshold": setting["scheduler_threshold"],

        "proxy_threshold" : trs["proxy_threshold"],
        "time_threshold" : trs["time_threshold"],
        "features": trs["features"], 
        
        "description": "EGAT model training run"
    }
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_or_setting=join("./checkpoints", configs["proxy_threshold"].__str__()+","+configs["time_threshold"].__str__()+","+configs["features"]["num_node_features"].__str__()+","+configs["features"]["num_edge_features"].__str__())
    os.makedirs(dir_or_setting, exist_ok=True)
    run_dir = join(dir_or_setting,f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    save_config(configs, join(run_dir, 'config.json'))

    all_files = [f for f in os.listdir(dir_of_batches) if f.endswith('_proc.pt')]
    
    random.seed(42)
    random.shuffle(all_files)
    
    # Split files
    total_files = len(all_files)
    n_train = int(0.8 * total_files)
    n_val = int(0.1 * total_files)
    
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train+n_val]
    test_files = all_files[n_train+n_val:] # TODO: test loop and if it is need at all

    # --- 1. Instantiate Datasets ---
    train_dataset = StrokeGraphDataset_class.StrokeGraphDataset(dir_of_batches, train_files)
    val_dataset = StrokeGraphDataset_class.StrokeGraphDataset(dir_of_batches, val_files)
    test_dataset = StrokeGraphDataset_class.StrokeGraphDataset(dir_of_batches, test_files)

    # --- 2. Instantiate Loaders ---
    batch_size = configs["batch_size"] 
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    save_config({
        "file train list:": train_dataset.file_list,
        "file val list:": val_dataset.file_list,
        "file test list:": test_dataset.file_list
                 }, join(run_dir, "file_list.json"))
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    sample_batch = next(iter(train_loader))
    model = EGAT_model(
        in_channels=sample_batch.x.size(1),
        hidden_channels=configs["hidden_channels"],
        out_channels=configs["out_channels"],
        edge_dim=sample_batch.edge_attr.size(1),
        heads=configs["heads"],
        num_hiden_layers=configs["hidden_layers"]
    ).to(device)

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

    early_stopper = EarlyStopper(patience=configs["early_stopper_patience"], path='./checkpoints/tmp_best_current_model.pt')
    
    for epoch in range(configs["epochs"]):
        start_time = time.time()

        # --- TRAINING LOOP ---
        model.train()
        epoch_train_loss = 0
        epoch_train_acc = 0
        train_count = 0
        
        for data in train_loader:
            data = data.to(device)
            
            data.train_mask = torch.ones(data.x.size(0), dtype=torch.bool, device=device)

            loss, acc = train_step(model, data, criterion, optimizer)
            
            epoch_train_loss += loss
            epoch_train_acc += acc
            train_count += 1

            del data 
            torch.cuda.empty_cache()

            
        # --- VALIDATION LOOP ---
        model.eval()
        epoch_val_loss = 0
        epoch_val_acc = 0
        val_count = 0
        
        for data in val_loader:
            data = data.to(device)

            # REPLACEMENT FOR set_batch_masks('val')
            data.val_mask = torch.ones(data.x.size(0), dtype=torch.bool, device=device)
            
            val_loss, val_acc = validate_step(model, data, criterion)
            
            epoch_val_loss += val_loss
            epoch_val_acc += val_acc
            val_count += 1

            del data 
            torch.cuda.empty_cache()
        
        # Avoid division by zero
        avg_train_loss = epoch_train_loss / train_count if train_count > 0 else 0
        avg_train_acc = epoch_train_acc / train_count if train_count > 0 else 0
        avg_val_loss = epoch_val_loss / val_count if val_count > 0 else 0
        avg_val_acc = epoch_val_acc / val_count if val_count > 0 else 0

        scheduler.step(avg_val_loss)
        early_stopper(avg_val_loss, model)

        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - start_time
        
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc, current_lr, epoch_time])

        if epoch % 1 == 0:
            print(f'Epoch {epoch:>3} | Train Loss: {avg_train_loss:.5f} | Train Acc: '
                  f'{avg_train_acc*100:>6.2f}% | Val Loss: {avg_val_loss:.5f} | '
                  f'Val Acc: {avg_val_acc*100:.2f}% | LR: {current_lr:.6f}')
            
        if (epoch) % 50 == 0 and epoch!=0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_val_loss,
                'config': configs
            }
            torch.save(checkpoint, join(run_dir, f'model_shot_{epoch}.pt'))
            
        # Clean memory at end of epoch
        gc.collect()
        torch.cuda.empty_cache()

        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch}!")
            model = early_stopper.load_best_model(model, device)
            break
        if os.path.exists(STOP_FILE):
            print("\nStop file detected! Breaking loop...")
            os.remove(STOP_FILE)
            break

    model.save_model(join(run_dir, 'final_model.pt'))
    
    # --- TEST LOOP ON TEST SET ---
    print("\nStarting Test Evaluation...")

    epoch_test_acc = 0
    test_count = 0
    for data in test_loader:
        data = data.to(device)

        data.test_mask = torch.ones(data.x.size(0), dtype=torch.bool, device=device)

        test_acc=test(model, data)

        epoch_test_acc+=test_acc
        test_count+=1

    
    avg_test_acc = epoch_test_acc / test_count if test_count > 0 else 0

    print(f"Final Test Accuracy: {avg_test_acc*100:.2f}%")
    
    save_config({"final_test_Accuracy":avg_test_acc},join(run_dir, 'final_acc.json'))

    return model

def _config_get_feature_names() -> dict:
    dummy_stroke = [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
    dummy_strokes = [dummy_stroke, dummy_stroke]

    feature = features.extract_stroke_features(dummy_strokes, 0, 1000.0, 1000.0)
 
    node_keys = list(feature["nodes"].keys())
    edge_keys = list(feature["edges_features"].keys())

    return {
        "node_features": node_keys,
        "edge_features": edge_keys,
        "num_node_features": len(node_keys),
        "num_edge_features": len(edge_keys)
    }

def pre_process_files(proxy_threshold:float, time_threshold:float)-> str:
    configs={
        "proxy_threshold":proxy_threshold,
        "time_threshold":time_threshold,
        "features": _config_get_feature_names()
    }
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = './batches/'
    
    all_files = [f for f in os.listdir(path) if (isfile(join(path, f)) and f.endswith('.pt') and not f.endswith('_proc.pt'))]
    path_of_procesed_filed=join(path, configs["proxy_threshold"].__str__()+","+configs["time_threshold"].__str__()+","+configs["features"]["num_node_features"].__str__()+","+configs["features"]["num_edge_features"].__str__())

    os.makedirs(path_of_procesed_filed, exist_ok=True)
    save_config(configs, join(path_of_procesed_filed, "setings.json"))

    files_to_process = [f for f in all_files if not os.path.exists(join(path_of_procesed_filed, f[:-3]+"_proc.pt"))]

    tasks = [(join(path, f), join(path_of_procesed_filed, f[:-3]+"_proc.pt"), device, configs['proxy_threshold'], configs['time_threshold']) for f in files_to_process]

    with Pool(processes=10, maxtasksperchild=1) as pool:
        pool.starmap(process_data_batch, tasks)
    
    print("All processes finished. Files created")
    return path_of_procesed_filed 

if __name__ == "__main__":
    pre_process_files(proxy_threshold=40.0, time_threshold=2.0)
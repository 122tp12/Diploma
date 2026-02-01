from calendar import c
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

def plot_strokes(strokes: dict, clasified, true_labels=None, save_path: str = None):
    """
    Plots strokes colored by classification.
    Assumes strokes are in absolute coordinates.
    """
    plt.figure(figsize=(6, 6))
    
    # Iterate through strokes
    for j in range(len(strokes)):          
        points = strokes[j]
        if len(points) < 1:
            continue

        # Extract coordinates directly (Absolute Coordinates)
        # Maintaining the (-x) transformation from the original code for orientation
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        if min(xs)<0 or min(ys)<0:
            print("Warning: Negative coordinates detected in stroke points.")
        # --- Determine Labels ---
        # Get Prediction
        pred_label = 0
        if isinstance(clasified, dict):
            pred_label = clasified.get(j, 0)
        else:
            if j < len(clasified):
                pred_label = clasified[j]

        # Get True Label (if available)
        real_label = None
        if true_labels is not None:
            if isinstance(true_labels, dict):
                real_label = true_labels.get(j, 0)
            else:
                if j < len(true_labels):
                    real_label = true_labels[j]

        color = 'red'
        linewidth = 1
        
        is_misclassified_special = False

        if real_label is not None:
            if real_label == 1 and pred_label != 1:
                # Missed text as light blue (cyan)
                color = 'cyan' 
                linewidth = 2
                is_misclassified_special = True
            elif real_label == 0 and pred_label != 0:
                # Missed non-text in orange
                color = 'orange'
                linewidth = 2
                is_misclassified_special = True
        
        if not is_misclassified_special:
            if pred_label == 1:
                color = 'blue'
                linewidth = 1
            elif pred_label == -1:
                color = 'green'
                linewidth = 5
            else:
                color = 'red'
                linewidth = 1
        
        plt.plot(xs, ys, color=color, linewidth=linewidth)
        
    plt.axis('equal')
    plt.gca().axis('off')
    plt.title("Render InkML (Pred)")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
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
        
        "description": setting["description"]
    }

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_or_setting=join("./checkpoints", configs["proxy_threshold"].__str__()+","+configs["time_threshold"].__str__()+","+configs["features"]["num_node_features"].__str__()+","+configs["features"]["num_edge_features"].__str__())
    if "augmentation_value" in trs:
        configs["augmentation_value"]=trs["augmentation_value"]
        dir_or_setting=join("./checkpoints", configs["proxy_threshold"].__str__()+","+configs["time_threshold"].__str__()+","+configs["features"]["num_node_features"].__str__()+","+configs["features"]["num_edge_features"].__str__()+","+configs["augmentation_value"].__str__())

    os.makedirs(dir_or_setting, exist_ok=True)
    run_dir = join(dir_or_setting,f'run_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    save_config(configs, join(run_dir, 'config.json'))

    missed_tests_dir = join(run_dir, "missed_tests")
    os.makedirs(missed_tests_dir, exist_ok=True)

    all_og_files = [f for f in os.listdir(dir_of_batches) if f.endswith('_proc.pt') and not f.startswith('aug')]
    
    random.seed(42)
    random.shuffle(all_og_files)
    
    # Split files
    total_files = len(all_og_files)
    n_train = int(0.8 * total_files)
    n_val = int(0.1 * total_files)
    
    train_files = all_og_files[:n_train]
    val_files = all_og_files[n_train:n_train+n_val]
    test_files = all_og_files[n_train+n_val:]

    # Add augmented files
    aug_files = []
    for f in train_files:
        base = f.replace('_proc.pt', '')
        aug_files.extend([af for af in os.listdir(dir_of_batches) if af.startswith('aug') and af.endswith('_proc.pt') and base in af])
    train_files.extend(aug_files)

    random.shuffle(train_files)
    
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
    
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([5.0, 1.0], device=device))

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

    # --- FAILED TESTS VISUALIZATION LOOP ---
    print("\nGenerating Failed Test Plots...")
    
    # We iterate manually to access raw strokes which are not in the graph tensors
    raw_data_dir = os.path.dirname(dir_of_batches) # Assuming raw batches are in the parent of processed dir
    
    model.eval()
    for file_name in test_files:
        # Load processed graph (for model)
        proc_path = join(dir_of_batches, file_name)
        data = torch.load(proc_path, map_location=device, weights_only=False)
        data = data.to(device)
        
        # Load raw data (for plotting)
        raw_file_name = file_name.replace('_proc.pt', '.pt')
        raw_path = join(raw_data_dir, raw_file_name)
        
        # Check if raw file exists (sanity check)
        if not os.path.exists(raw_path):
            print(f"Warning: Raw file {raw_path} not found. Skipping plot for {file_name}")
            continue

        raw_strokes_list, raw_labels_list = _load_batch(raw_path, device='cpu')

        # Run model
        with torch.no_grad():
            out = model(data.x, data.edge_index, data.edge_attr) 
            pred = out.argmax(dim=1)  # Predictions for all nodes in this batch

        # The processed 'data' object is a concatenation of multiple graphs (strokes/characters).
        # We need to map the flat prediction tensor back to individual raw graphs.
        
        current_idx = 0
        for i, (strokes, true_labels) in enumerate(zip(raw_strokes_list, raw_labels_list)):
            # Calculate how many nodes (strokes) are in this specific graph
            # We can infer this from the extracted features, but we don't have them here easily.
            # Alternatively, we rely on the fact that true_labels length = num_nodes
            
            num_nodes = len(true_labels)
            
            # Slice the predictions for this specific graph
            graph_pred = pred[current_idx : current_idx + num_nodes].cpu().tolist()
            graph_true = true_labels # already a list
            
            current_idx += num_nodes
            
            # Check if prediction matches truth
            if graph_pred != graph_true:
                # Mismatch found! Plot this specific graph.
                
                # Plot with PREDICTED labels AND TRUE labels for color logic
                plot_filename = f"fail_{raw_file_name[:-3]}_idx{i}.png"
                save_path = join(missed_tests_dir, plot_filename)
                
                plot_strokes(strokes, graph_pred, true_labels=graph_true, save_path=save_path)
        
        # Cleanup
        del data
        torch.cuda.empty_cache()

    print(f"Failed test plots saved to: {missed_tests_dir}")

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
    if any(s for s in all_files if s.startswith("aug")):
        aug_val=max([int(s[3:4]) for s in all_files if s.startswith("aug")])+1
        path_of_procesed_filed=join(path, configs["proxy_threshold"].__str__()+","+configs["time_threshold"].__str__()+","+configs["features"]["num_node_features"].__str__()+","+configs["features"]["num_edge_features"].__str__()+","+aug_val.__str__())
        configs["augmentation_value"]=aug_val
    else:
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
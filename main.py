from genericpath import isfile
import os
from posixpath import join
from turtle import st
import matplotlib.pyplot as plt
from typing import List, Optional
from sympy import true
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import datetime

import torch

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

def _load_batch(path, device)-> tuple[List[dict], List[dict]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt['strokes'], ckpt['labels']

def _save_data_batch(data: Data, path: str):
    torch.save(data.to(torch.device('cpu')), path)

def load_data_batch(path: str, device) -> Data:
    data = torch.load(path, map_location=device, weights_only=False)
    return data

def process_data_batch(batch: str, device) -> Data:
    if os.path.exists(batch[:-3]+"_proc.pt"):
        return load_data_batch(batch[:-3]+"_proc.pt", device=device)

    strokes_dict, true_y_dict=_load_batch(batch, device)

    x= torch.empty((0,8), dtype=torch.float)
    edge_index=torch.empty((2,0), dtype=torch.long)
    edge_attr=torch.empty((0,6), dtype=torch.float)
    true_y=torch.empty((0,), dtype=torch.long)

    offset=0
    for i in range(strokes_dict.__len__()):
        stroke=strokes_dict[i]
        true_y_part=true_y_dict[i]
        stroke_list=[]
        for j in list(stroke.keys()):
            stroke_list.append(stroke[j])

        dict_features=features.extract_stroke_features(stroke_list, offset=offset) # TODO: change from stroke_list to stroke
        x_tmp=torch.tensor(list(zip(
                dict_features["nodes"]["length"],
                dict_features["nodes"]["width_height_ratio"],
                dict_features["nodes"]["stroke_area"],
                dict_features["nodes"]["straightness"],
                dict_features["nodes"]["num_spatial_neighbors"],
                dict_features["nodes"]["duration"],
                dict_features["nodes"]["pca_ratio"],
                dict_features["nodes"]["accumulated_curvature"],
                )), dtype=torch.float)
        edge_index_tmp=torch.tensor(dict_features["edge_index"], dtype=torch.long).t().contiguous()
        edge_attt_tmp=torch.tensor(list(zip(
                dict_features["edges_features"]["min_distance"],
                dict_features["edges_features"]["min_endpoint_distance"],
                dict_features["edges_features"]["centroid_distance"],
                dict_features["edges_features"]["direction_cosine"],
                dict_features["edges_features"]["temporal_distance"],
                dict_features["edges_features"]["centroid_dx"],
                )), dtype=torch.float)
        true_y_tmp=torch.tensor([true_y_part[k] for k in stroke.keys()], dtype=torch.long)


        x=torch.cat((x, x_tmp), dim=0)
        edge_index=torch.cat((edge_index, edge_index_tmp), dim=1)
        edge_attr=torch.cat((edge_attr, edge_attt_tmp), dim=0)
        true_y=torch.cat((true_y, true_y_tmp), dim=0)

        offset+=stroke.__len__()

    del strokes_dict
    del true_y_dict

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

def main_train_loop(device)-> tuple[List[int], EGAT_model]:
    out_channels = 2   # Кількість класів
    hidden_channels = 20
    hidden_layers = 2
    heads = 4
    
    path = './batches/'
    files = [f for f in os.listdir(path) if f.endswith('_proc.pt')]

    temp_data = load_data_batch(join(path, files[0]), device)

    model = EGAT_model(
        in_channels=temp_data.x.size(1),
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        edge_dim=temp_data.edge_attr.size(1),
        heads=heads,
        num_hiden_layers=hidden_layers
    ).to(device)
    
    del temp_data 
    torch.cuda.empty_cache()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
            mode='min', 
            factor=0.5,
            patience=10,
            threshold=0.0001
            )
    
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0], device=device))

    global_train_acc=[]
    global_val_acc=[]

    early_stopper = EarlyStopper(patience=150, path='./checkpoints/tmp_best_current_model.pt')
    epochs = 1000
    for epoch in range(epochs):
        epoch_train_loss = 0
        epoch_val_loss = 0
        epoch_train_acc = 0
        epoch_val_acc = 0
        count = 0
        
        # --- Цикл по файлах (завантажив -> навчив -> видалив) ---
        for file_name in files:
            # А. Завантаження
            full_path = join(path, file_name)
            data = load_data_batch(full_path, device)
            
            # Б. Крок навчання
            loss, acc = train_step(model, data, criterion, optimizer)
            
            # В. Крок валідації (одразу на цьому ж графі)
            val_loss, val_acc = validate_step(model, data, criterion)
            
            # Г. Статистика
            epoch_train_loss += loss
            epoch_train_acc += acc
            
            epoch_val_loss += val_loss
            epoch_val_acc += val_acc
            count += 1
            
            del data
        
        avg_train_loss = epoch_train_loss / count
        avg_train_acc = epoch_train_acc / count
        avg_val_loss = epoch_val_loss / count
        avg_val_acc = epoch_val_acc / count

        scheduler.step(avg_val_loss)
        early_stopper(avg_val_loss, model)

        current_lr = optimizer.param_groups[0]['lr']
        if epoch % 1 == 0:
            print(f'Epoch {epoch:>3} | Train Loss: {avg_train_loss:.5f} | Train Acc: '
                  f'{avg_train_acc*100:>6.2f}% | Val Loss: {avg_val_loss:.5f} | '
                  f'Val Acc: {avg_val_acc*100:.2f}% | LR: {current_lr:.6f}')
        
        global_train_acc.append(avg_train_acc)
        global_val_acc.append(avg_val_acc)
        
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch}!")
            model=early_stopper.load_best_model(model, device)
            return model
    

    plt.plot(global_train_acc, global_val_acc)

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train and validation accuracy")

    plt.show()


    model.save_model(f'./checkpoints/model_chekpoint_{datetime.datetime.now().timestamp()}.pt')

    

    return model


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = './batches/'
    files = [f for f in os.listdir(path) if (isfile(join(path, f)) and f.endswith('.pt') and not f.endswith('_proc.pt'))]

    for f in files:
        strokes, true_y = _load_batch(path+f, device)
        i=0
        for i in range(strokes.__len__()):
            if true_y[i].keys()!=strokes[i].keys():
                print("Mismatch between strokes and labels keys in file:", f, "at index:", i)
                
        if not os.path.exists(join(path, f[:-3]+"_proc.pt")):
            print(f"Processing {f}...")
            process_data_batch(join(path, f), device)
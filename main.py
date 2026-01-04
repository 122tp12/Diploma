from genericpath import isfile
import os
from posixpath import join
import matplotlib.pyplot as plt
from typing import List, Optional
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import datetime

import torch

from EGAT import EGAT_model, train
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

def load_batch(path, device)-> tuple[List[dict], List[dict]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt['strokes'], ckpt['labels']

def save_data_batch(data: Data, path: str):
    torch.save(data.to(torch.device('cpu')), path)

def load_data_batch(path: str, device) -> Data:
    data = torch.load(path, map_location=device, weights_only=False)
    return data

def process_data_batch(strokes_dict: List[dict], true_y_dict: List[dict], batch: str, device) -> Data:
    if os.path.exists(batch[:-3]+"_proc.pt"):
        return load_data_batch(batch[:-3]+"_proc.pt", device=device)

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

        dict_features=features.extract_stroke_features(stroke_list, offset=offset)
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
        true_y_tmp=torch.tensor([true_y_part['t'+k.__str__()] for k in range(len(stroke_list))], dtype=torch.long)


        x=torch.cat((x, x_tmp), dim=0)
        edge_index=torch.cat((edge_index, edge_index_tmp), dim=1)
        edge_attr=torch.cat((edge_attr, edge_attt_tmp), dim=0)
        true_y=torch.cat((true_y, true_y_tmp), dim=0)

        offset+=stroke_list.__len__()

    data=Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=true_y,
        )
    
    scaler = StandardScaler()
    data.x = torch.from_numpy(scaler.fit_transform(data.x)).float() # type: ignore
    data.edge_attr = torch.from_numpy(scaler.fit_transform(data.edge_attr)).float()# type: ignore

    data.train_mask, data.val_mask = features.get_masks(true_y.__len__())


    data.x = data.x.to(device)
    data.edge_attr = data.edge_attr.to(device)
    data.edge_index = data.edge_index.to(device)# type: ignore
    data.y = data.y.to(device)# type: ignore
    data.train_mask = data.train_mask.to(device)
    data.val_mask = data.val_mask.to(device)#TODO: cheak if works

    save_data_batch(data, batch[:-3]+"_proc.pt")
    return data

def train_model(data: Data, device, model: Optional[EGAT_model]=None)-> tuple[List[int], EGAT_model]:
    out_channels = 2   # Кількість класів
    hidden_channels = 20
    
    in_channels = data.x.size(1)# type: ignore
    edge_dim = data.edge_attr.size(1)# type: ignore

    if model==None:
        model = EGAT_model(in_channels, hidden_channels, out_channels, edge_dim)
        model = model.to(device)
        # recreate optimizer so optimizer state lives on same device
        model.optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        
    model=train(model, data)
    out = model(data.x, data.edge_index, data.edge_attr)

    out=out.argmax(dim=1).to(torch.device('cpu')).numpy().tolist()
    
    model.save_model(f'./checkpoints/model_chekpoint_{datetime.datetime.now().timestamp()}.pt')

    return out, model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = './batches/'
files = [f for f in os.listdir(path) if (isfile(join(path, f)) and f.endswith('.pt') and not f.endswith('_proc.pt'))]
model=None

for file in files:
    strokes, true_y = load_batch(path+file, device)
    i=0
    for i in range(strokes.__len__()):
        if list(set(true_y[i].values())).__len__()==3 or true_y[i].__len__()!=strokes[i].__len__():
            plot_strokes(strokes[i], true_y[i])

"""device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

strokes, true_y = load_batch('batches/batch0.pt', device)
data=process_data_batch(strokes, true_y, batch='data_batches/data_batch0.pt')
data = data.to(device) 
clasified, model=train_model(data, device)
plot_strokes(strokes[0], true_y[0])
plot_strokes(strokes[0], clasified)"""

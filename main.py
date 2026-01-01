import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy
from typing import List, Optional
from torch_geometric.data import Data
import torch_geometric.transforms as T
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn.functional as F

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
                plt.plot(xs, ys, color='green', linewidth=1)
            else:
                plt.plot(xs, ys, color='red', linewidth=1)
        else:
            if clasified[j]==1:
                plt.plot(xs, ys, color='blue', linewidth=1)
            elif clasified[j]==-1:
                plt.plot(xs, ys, color='green', linewidth=1)
            else:
                plt.plot(xs, ys, color='red', linewidth=1)
            
        
        j+=1

    plt.axis('equal')
    plt.gca().axis('off')
    plt.title("Render InkML")
    plt.show()

def load_batch(path, device='cpu')-> tuple[dict, dict]:
    ckpt = torch.load(path, map_location=device)
    return ckpt['strokes'], ckpt['labels']

def train_model(strokes_dict: dict, true_y_dict: dict, model: Optional[EGAT_model]=None)-> tuple[List[int], EGAT_model]:
    out_channels = 2   # Кількість класів
    hidden_channels = 30
    strokes=[]
    true_y=[]
    
    
    for i in list(strokes_dict.keys()):
        strokes.append(strokes_dict[i])
        true_y.append(true_y_dict[i])
    

    dict_features=features.extract_stroke_features(strokes)
    in_channels = dict_features["nodes"].__len__()  # Кількість ознак вузла
    edge_dim = dict_features["edges_features"].__len__()    # Розмірність ознак ребра
    
    data=Data(
        x=torch.tensor(list(zip(
            dict_features["nodes"]["length"],
            dict_features["nodes"]["width_height_ratio"],
            dict_features["nodes"]["stroke_area"],
            dict_features["nodes"]["straightness"],
            dict_features["nodes"]["num_spatial_neighbors"],
            dict_features["nodes"]["duration"],
            dict_features["nodes"]["pca_ratio"],
            dict_features["nodes"]["accumulated_curvature"],
            )), dtype=torch.float),
        edge_index=torch.tensor(dict_features["edge_index"], dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(list(zip(
            dict_features["edges_features"]["min_distance"],
            dict_features["edges_features"]["min_endpoint_distance"],
            dict_features["edges_features"]["centroid_distance"],
            dict_features["edges_features"]["direction_cosine"],
            dict_features["edges_features"]["temporal_distance"],
            dict_features["edges_features"]["centroid_dx"],
            
            )), dtype=torch.float),
        y=torch.tensor(true_y, dtype=torch.long),
        
    )

    scaler = StandardScaler()
    data.x = torch.from_numpy(scaler.fit_transform(data.x)).float()
    data.edge_attr = torch.from_numpy(scaler.fit_transform(data.edge_attr)).float()

    data.train_mask, data.val_mask = features.get_masks(true_y.__len__())
    if model==None:
        model = EGAT_model(in_channels, hidden_channels, out_channels, edge_dim)
    model=train(model, data)
    out = model(data.x, data.edge_index, data.edge_attr)

    out=out.argmax(dim=1).numpy().tolist()

    return out, model

print(torch.cuda.is_available())

strokes, true_y = load_batch('batches/batch0.pt', device='cpu')

clasified, model=train_model(strokes[0], true_y[0])#TODO: full batch
plot_strokes(strokes[0], true_y[0])
plot_strokes(strokes[0], clasified)
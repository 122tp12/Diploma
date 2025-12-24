import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy
from typing import List
import features

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# To save dependencies:
# pip freeze > requirements.txt

class EGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_dim, heads=4):
        super(EGAT, self).__init__()
        
        # Перший шар GAT
        # edge_dim — це ключовий параметр, який перетворює GAT на EGAT
        self.conv1 = GATConv(
            in_channels=in_channels, 
            out_channels=hidden_channels, 
            heads=heads, 
            edge_dim=edge_dim,  # <--- Вказуємо розмірність ознак ребер
            concat=True
        )
        
        # Другий шар GAT (вихідний)
        self.conv2 = GATConv(
            in_channels=hidden_channels * heads, 
            out_channels=out_channels, 
            heads=1, 
            edge_dim=edge_dim,  # <--- Тут також враховуємо ребра
            concat=False
        )

    def forward(self, x, edge_index, edge_attr):
        """
        x: [num_nodes, in_channels] - ознаки вузлів
        edge_index: [2, num_edges] - топологія графа
        edge_attr: [num_edges, edge_dim] - ознаки ребер
        """
        
        # 1. Перший шар згортки
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        
        # 2. Другий шар згортки
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        
        return F.log_softmax(x, dim=1)

def plot_strokes(strokes: List[List[List[float]]]):
    for points in strokes:
    
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
        
        plt.plot(xs, ys, color=numpy.random.rand(3,), linewidth=1)

    plt.axis('equal')
    plt.gca().axis('off')
    plt.title("Render InkML")
    plt.show()

def parse_inkml(file_path: str) -> List[List[List[float]]]:
    tree = ET.parse(file_path)

    root = tree.getroot()

    traces = root.findall('.//{http://www.w3.org/2003/InkML}trace')
    if not traces:
        traces = root.findall('.//trace')

    if not traces:
        print("No traces found.")
        raise ValueError("No traces found in the InkML file.")
    
    strokes = []
    for trace in traces:
        tmp=[]
        if trace.text is None:
            continue
        for sub_trace in trace.text.strip().split(','):
            if "\'" in sub_trace :
                clean_text = sub_trace.replace("'", " ").strip()
                parts = clean_text.split(' ')
            elif '\"' in sub_trace:
                clean_text = sub_trace.replace('"', " ").strip()
                parts = clean_text.split(' ')
            else:
                clean_text=sub_trace.replace('-', " -").strip()
                parts= clean_text.split(' ')
                
            tmp.append([float(p) for p in parts])
        strokes.append(tmp)
    
    return strokes

def train_model():
    num_nodes = 100
    num_edges = 300
    in_channels = 2   # Розмірність ознак вузла
    edge_dim = 1       # Розмірність ознак ребра
    out_channels = 2   # Кількість класів
    hidden_channels = 8

    dict_features=features.extract_stroke_features(strokes)
    x=torch.tensor(list(zip(dict_features["nodes"]["length"],dict_features["nodes"]["width_height_ratio"])), dtype=torch.float)
    edge_index=torch.tensor(dict_features["edge_index"], dtype=torch.long).t().contiguous()
    edge_attr=torch.tensor(list(zip(dict_features["edges_features"]["min_distance"])), dtype=torch.float)

    model = EGAT(in_channels, hidden_channels, out_channels, edge_dim)

    out = model(x, edge_index, edge_attr)

    print("Розмірність виходу:", out.shape)
    print("Успішний прохід!")

strokes = parse_inkml('./IAMonDo-db-1.0/001.inkml')
train_model()
plot_strokes(strokes)
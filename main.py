import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy
from typing import List
from torch_geometric.data import Data

import torch
import torch.nn.functional as F

from EGAT import EGAT_model, train
import features
# To save dependencies:
# pip freeze > requirements.txt

def plot_strokes(strokes: List[List[List[float]]], clasified):
    i=0
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
        
        plt.plot(xs, ys, color=clasified[i], linewidth=1)
        i+=1

    plt.axis('equal')
    plt.gca().axis('off')
    plt.title("Render InkML")
    plt.show()

def parse_inkml(file_path: str) -> tuple[List[List[List[float]]], List[int]]:
    def get_labeled_strokes(root):
        """
    Приймає root XML-дерева.
    Повертає список словників: [{'id': 't1', 'label': 1}, ...]
    де 1 - Текст, 0 - Не текст.
        """
        dataset = []
    
        # Визначаємо множини для типів (можна розширювати)
        TEXT_TYPES = {'Textblock', 'Textline', 'Word', 'Correction'}
        NONTEXT_TYPES = {'Diagram', 'Drawing', 'Marking', 'Marking_Underline', 
                     'Marking_Angle', 'Marking_Bracket', 'Marking_Sideline'}

        def _walk(node, current_label):
            """Внутрішня функція для обходу дерева зі збереженням контексту"""
            new_label = current_label
        
            # 1. Шукаємо анотацію типу у поточному вузлі (через ітерацію, щоб ігнорувати namespaces)
            for child in node:
                if child.tag.endswith('annotation') and child.attrib.get('type') == 'type':
                    ann_text = child.text
                    if ann_text in TEXT_TYPES:
                        new_label = 1 # Клас: Текст
                    elif ann_text in NONTEXT_TYPES:
                        new_label = 0 # Клас: Не текст
        
            # 2. Якщо це штрих (є посилання traceDataRef), записуємо результат
            ref = node.attrib.get('traceDataRef')
            if ref:
                # Видаляємо '#' з ID та додаємо в масив
                clean_id = ref.replace('#', '')
                dataset.append(new_label)
                # Штрих - це лист дерева, далі йти не треба
                return

            # 3. Якщо це контейнер (traceView) - йдемо вглиб (рекурсія)
            # Ми не використовуємо тут findall('.//'), щоб не перескочити рівні вкладеності
            for child in node:
                if child.tag.endswith('traceView'):
                    _walk(child, new_label)

        # Точка входу: шукаємо кореневі traceView
        for child in root:
            if child.tag.endswith('traceView'):
                _walk(child, -1) # -1 означає "невідомий тип" (на випадок помилки розмітки)

        return dataset
    
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
    
    true_y=get_labeled_strokes(root)
    return (strokes, true_y)

def train_model(strokes: List[List[List[float]]], true_y: List[int]):
    num_nodes = 100
    num_edges = 300
    in_channels = 2   # Розмірність ознак вузла
    edge_dim = 1       # Розмірність ознак ребра
    out_channels = 2   # Кількість класів
    hidden_channels = 8

    dict_features=features.extract_stroke_features(strokes)
    data=Data(
        x=torch.tensor(list(zip(dict_features["nodes"]["length"],dict_features["nodes"]["width_height_ratio"])), dtype=torch.float),
        edge_index=torch.tensor(dict_features["edge_index"], dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(list(zip(dict_features["edges_features"]["min_distance"])), dtype=torch.float),
        y=torch.tensor(true_y, dtype=torch.long),
        
    )
    data.train_mask, data.val_mask = features.get_masks(true_y.__len__())

    model = EGAT_model(in_channels, hidden_channels, out_channels, edge_dim)
    model=train(model, data)
    out = model(data.x, data.edge_index, data.edge_attr)

    out=out.argmax(dim=1).numpy().tolist()

    #print("Розмірність виходу:", out.shape)
    print("Успішний прохід!")
    return out

strokes, true_y = parse_inkml('./IAMonDo-db-1.0/001.inkml')
clasified=train_model(strokes, true_y)
plot_strokes(strokes, clasified)
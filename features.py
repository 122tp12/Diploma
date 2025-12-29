import numpy as np
import torch
type list_strokes = list[list[list[float]]]

# Utility functions to create edge by proximity and and time
def get_edges(strokes: list_strokes, threshold: float = 50.0) -> list[tuple[int, int]]:
    edges = []

    for i in range(len(strokes)):
        for j in range(len(strokes)):
            if i == j:
                continue
            if i-1==j or i+1==j:
                edges.append((i, j))
                continue
            for point1 in strokes[i]:
                for point2 in strokes[j]:
                    distance = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
                    if distance < threshold:
                        edges.append((i, j))
                        break
                else:
                    continue
                break

    return edges

# Node features:
def stroke_length(strokes: list_strokes) -> list[float]:
    length = []

    for stroke in strokes:
        if len(strokes) < 2:
            length.append(0.0)
            continue
        for i in range(len(stroke) - 2):
            x1, y1 = stroke[i][0], stroke[i][1]
            x2, y2 = stroke[i + 1][0], stroke[i + 1][1]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            length.append(distance)

    
    return length

def width_height_ratio(strokes: list_strokes) -> list[float]:
    ratio = []

    for stroke in strokes:
        width = max(point[0] for point in stroke) - min(point[0] for point in stroke)
        height = max(point[1] for point in stroke) - min(point[1] for point in stroke)
        if len(stroke) < 1:
            ratio.append(0.0)
            continue
        if height == 0:
            ratio.append(0.0)
            continue
        ratio.append(width / height)
    
    return ratio

# Edge features:
def min_distance_between_strokes(strokes: list_strokes, edges:list[tuple[int, int]]) -> list[float]:
    distances = []

    for edge in edges:
        stroke1 = strokes[edge[0]]
        stroke2 = strokes[edge[1]]
        min_distance = float('inf')

        for point1 in stroke1:
            for point2 in stroke2:
                distance = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
                if distance < min_distance:
                    min_distance = distance

        distances.append(min_distance)

    return distances

# Main function to extract all features
def extract_stroke_features(strokes: list_strokes) -> dict:
    edges=get_edges(strokes, threshold=10.0)
    
    return {
        "nodes":{
            "length": stroke_length(strokes),
            "width_height_ratio": width_height_ratio(strokes)
        },
        "edge_index": edges,
        "edges_features": {
            "min_distance": min_distance_between_strokes(strokes, edges)
        }
    }

def get_masks(num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    perm = torch.randperm(num_nodes)

    separ = int(0.75 * num_nodes)

    train_idx = perm[:separ]
    val_idx = perm[separ:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True

    return (train_mask, val_mask)
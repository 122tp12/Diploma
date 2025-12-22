import numpy as np
type list_strokes = list[list[list[float]]]

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


def extract_stroke_features(strokes: list_strokes) -> dict:
    return {
        "length": stroke_length(strokes),
        "width_height_ratio": width_height_ratio(strokes)
    }
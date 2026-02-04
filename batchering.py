from os import listdir
from os.path import isfile, join
import random
import math
import xml.etree.ElementTree as ET
from scipy.fftpack import shift
import torch


mypath = './IAMonDo-db-1.0/'

uniqe_types = []
def parse_inkml(file_path: str):

    def get_canvas_transform(root):

        try:

            canvas_transform = root.find('.//{http://www.w3.org/2003/InkML}canvasTransform')
            if canvas_transform is None:
                canvas_transform = root.find('.//canvasTransform')
            
            if canvas_transform is not None:
                matrix_elem = canvas_transform.find('.//{http://www.w3.org/2003/InkML}matrix')
                if matrix_elem is None:
                    matrix_elem = canvas_transform.find('.//matrix')
                
                if matrix_elem is not None and matrix_elem.text:
                    matrix_text = matrix_elem.text.strip()
                    rows = []
                    for row_str in matrix_text.split(','):
                        row_str = row_str.strip()
                        if row_str:
                            row = [float(x) for x in row_str.split()]
                            rows.append(row)
                    
                    if len(rows) >= 2 and len(rows[0]) >= 3:
                        return rows
        except Exception as e:
            print(f"Warning: Could not parse canvas transform: {e}")
        
        return None
    
    def apply_canvas_transform(points, matrix):
        """Застосовує афінну трансформацію до координат"""
        if matrix is None or len(matrix) < 2:
            return points
        
        transformed = []
        for x, y, *rest in points:
            # Афінна трансформація: x' = m[0][0]*x + m[0][1]*y + m[0][4]
            #                        y' = m[1][0]*x + m[1][1]*y + m[1][4]
            new_x = matrix[0][0] * x + matrix[0][1] * y + (matrix[0][4] if len(matrix[0]) > 4 else 0)
            new_y = matrix[1][0] * x + matrix[1][1] * y + (matrix[1][4] if len(matrix[1]) > 4 else 0)
            
            if rest:
                transformed.append((new_x, new_y, *rest))
            else:
                transformed.append((new_x, new_y))
        
        return transformed

    def get_labeled_strokes(root) -> dict:
        dataset = {}
    
        TEXT_TYPES = {'Textblock', 'Textline', 'Word', 'Correction'}
        NONTEXT_TYPES = {
            'Drawing', 'Diagram', 'Arrow',                  # Графіка
            'Formula', 'Symbol',                            # Математика/Символи
            'Table', 'List', 'Structure',                   # Структура
            'Marking', 'Marking_Encircling',                # Розмітка...
            'Marking_Underline', 'Marking_Sideline', 
            'Marking_Bracket', 'Marking_Angle', 
            'Marking_Connection',
            'Document',
            'Garbage'
        }
        IGNORE_TYPES = {'Document'}

        def _walk(node, current_label):
            """Внутрішня функція для обходу дерева зі збереженням контексту"""
            new_label = current_label
             
            for child in node:
                if child.tag.endswith('annotation') and child.attrib.get('type') == 'type':
                    ann_text = child.text
                    if ann_text not in uniqe_types:
                        uniqe_types.append(ann_text)
                    if ann_text in TEXT_TYPES:
                        new_label = 1 # Клас: Текст
                    elif ann_text in NONTEXT_TYPES:
                        new_label = 0 # Клас: Не текст
        
            ref = node.attrib.get('traceDataRef')
            if ref:
                
                dataset[ref[1:]]=new_label
                return

            for child in node:
                if child.tag.endswith('traceView'):
                    _walk(child, new_label)

        for child in root:
            if child.tag.endswith('traceView'):
                _walk(child, -1)

        return dataset
    
    def to_absolute_coords_and_y_align(strokes: dict, true: dict):
        
        strokes_list=[]
        true_list=[]
        for j in strokes.keys():
            points=strokes[j]
            current_x = points[0][0]
            current_y = points[0][1]
            current_t=points[0][2]
            current_f=points[0][3]
            stroke_points = [(-current_x, current_y, current_t)] # [(-current_x, current_y, current_t, current_f)]
            if len(points)<2: # skip points to propper features
                continue

            true_list.append(true[j])

            i=2
            vx, vy, vt, vf = points[1][0], points[1][1], points[1][2], points[1][3]

            current_x += vx
            current_y += vy
            current_t += vt
            current_f += vf
            stroke_points.append((-current_x, current_y, current_t))#stroke_points.append((-current_x, current_y, current_t, current_f))

            while i < len(points):
                vx += points[i][0]
                vy += points[i][1]
                vt += points[i][2]
                vf += points[i][3]
                
                current_x += vx
                current_y += vy
                current_t += vt
                current_f += vf
                stroke_points.append((-current_x, current_y, current_t))#stroke_points.append((-current_x, current_y, current_t, current_f))
                
                i+=1
            strokes_list.append(stroke_points)

        shift_x=0
        shift_y=0
        for stroke in strokes_list:
            for point in stroke:
                if shift_x>point[0]:
                    shift_x=point[0]
                if shift_y>point[1]:
                    shift_y=point[1]
        
        shifted_strokes_list=[]
        for stroke in strokes_list:
            shifted_stroke=[]
            for point in stroke:
                shifted_stroke.append((point[0]-shift_x, point[1]-shift_y, point[2]))
            shifted_strokes_list.append(shifted_stroke)
                
        

        return shifted_strokes_list, true_list

    tree = ET.parse(file_path)

    root = tree.getroot()

    canvas_matrix = get_canvas_transform(root)

    traces = root.findall('.//{http://www.w3.org/2003/InkML}trace')
    if not traces:
        traces = root.findall('.//trace')

    if not traces:
        print("No traces found.")
        raise ValueError("No traces found in the InkML file.")
    
    strokes = {}
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
        
        if canvas_matrix is not None:
            tmp = apply_canvas_transform(tmp, canvas_matrix)
        
        strokes[trace.attrib[list(trace.attrib.keys())[0]]]=tmp
    
    true_y=get_labeled_strokes(root)
    
    strokes, true_y=to_absolute_coords_and_y_align(strokes, true_y)

    return (strokes, true_y)

def save_batch(path, strokes_batch, labels_batch):
    torch.save({'strokes': strokes_batch, 'labels': labels_batch}, path)

def rotate(points, angle_deg):
    if not points:
        return points

    # Determine if input is a single stroke or list of strokes
    is_single = False
    first = points[0]
    if isinstance(first, (int, float)) or (isinstance(first, (list, tuple)) and isinstance(first[0], (int, float))):
        # Looks like a single stroke (first element is a point)
        strokes = [points]
        is_single = True
    else:
        strokes = points

    # Collect all coordinates to compute global centroid
    xs = []
    ys = []
    for stroke in strokes:
        for p in stroke:
            xs.append(p[0])
            ys.append(p[1])

    if not xs:
        return points

    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)

    if angle_deg>=0:
        angle = -math.radians(angle_deg)
    else:
        angle = math.radians(-angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    rotated_strokes = []
    for stroke in strokes:
        new_stroke = []
        for x, y, *rest in stroke:
            dx = x - cx
            dy = y - cy
            rx = dx * cos_a - dy * sin_a + cx
            ry = dx * sin_a + dy * cos_a + cy
            if rest:
                new_stroke.append((rx, ry, *rest))
            else:
                new_stroke.append((rx, ry))
        rotated_strokes.append(new_stroke)

    # Ensure all coordinates are non-negative
    min_x = min(p[0] for s in rotated_strokes for p in s)
    min_y = min(p[1] for s in rotated_strokes for p in s)
    shift_x = -min_x if min_x < 0 else 0.0
    shift_y = -min_y if min_y < 0 else 0.0

    if shift_x != 0.0 or shift_y != 0.0:
        shifted = []
        for s in rotated_strokes:
            ns = []
            for x, y, *rest in s:
                rx = x + shift_x
                ry = y + shift_y
                if rest:
                    ns.append((rx, ry, *rest))
                else:
                    ns.append((rx, ry))
            shifted.append(ns)
        rotated_strokes = shifted

    return rotated_strokes[0] if is_single else rotated_strokes

def scale(points, scale_factor_x, scale_factor_y):
    scaled_strokes = []
    for stroke in points:
        new_stroke = []
        for x, y, t in stroke:
            sx = x * scale_factor_x
            sy = y * scale_factor_y
            new_stroke.append((sx, sy, t))
        scaled_strokes.append(new_stroke)
    return scaled_strokes

onlyfiles = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.inkml'))]

for file in onlyfiles:
    strokes_batch = []
    true_y_batch = []
    strokes, true_y = parse_inkml(join(mypath, file))
    strokes_batch.append(strokes)
    true_y_batch.append(true_y)
    save_batch(f'batches/batch{file[:-6]}.pt', strokes_batch, true_y_batch)

    #Augmentations
    for i in range(3):
        strokes_rotated = rotate(strokes, random.uniform(-15.0, 15.0))
        strokes_scaled = scale(strokes_rotated, random.uniform(0.8, 1.2), random.uniform(0.8, 1.2))
        strokes_batch = []
        strokes_batch.append(strokes_scaled)
        save_batch(f'batches/aug{i}_batch{file[:-6]}.pt', strokes_batch, true_y_batch)

print(uniqe_types)
from os import listdir, makedirs
from os.path import isfile, join
from pydoc import text
import random
import math
import xml.etree.ElementTree as ET
from scipy.fftpack import shift
import torch

mypath = './IAMonDo-db-1.0/'
uniqe_types = []

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
    if matrix is None or len(matrix) < 2:
        return points
    transformed = []
    for p in points:
        x, y = p[0], p[1]
        new_x = matrix[0][0] * x + matrix[0][1] * y + (matrix[0][4] if len(matrix[0]) > 4 else 0)
        new_y = matrix[1][0] * x + matrix[1][1] * y + (matrix[1][4] if len(matrix[1]) > 4 else 0)
        if len(p) > 2:
            transformed.append((new_x, new_y, *p[2:]))
        else:
            transformed.append((new_x, new_y))
    return transformed

def get_labeled_strokes(root) -> dict:
    global uniqe_types
    dataset = {}
    TEXT_TYPES = {'Textblock', 'Textline', 'Word', 'List', 'Symbol'}
    NONTEXT_TYPES = {
        'Drawing', 'Diagram', 'Arrow', 'Formula',
        'Table', 'Structure', 'Marking', 'Marking_Encircling',
        'Correction', 'Marking_Underline', 'Marking_Sideline', 'Marking_Bracket', 
        'Marking_Angle', 'Marking_Connection', 'Document', 'Garbage'
    }


    def _walk(node, current_label):
        new_label = current_label
        for child in node:
            if child.tag.endswith('annotation') and child.attrib.get('type') == 'type':
                ann_text = child.text
                if ann_text not in uniqe_types:
                    uniqe_types.append(ann_text)
                if ann_text in TEXT_TYPES:
                    new_label = 1
                elif ann_text in NONTEXT_TYPES:
                    new_label = 0
    
        ref = node.attrib.get('traceDataRef')
        if ref:
            dataset[ref[1:]] = new_label
            return

        for child in node:
            if child.tag.endswith('traceView'):
                _walk(child, new_label)

    for child in root:
        if child.tag.endswith('traceView'):
            _walk(child, -1)
    return dataset

def to_absolute_coords_and_y_align(strokes: dict, true: dict):
    strokes_list = []
    true_list = []
    for j in strokes.keys():
        if j not in true: continue # Пропускаємо штрихи без міток
        points = strokes[j]
        if len(points) < 2: continue

        current_x = points[0][0]
        current_y = points[0][1]
        current_t = points[0][2] if len(points[0]) > 2 else 0
        current_f = points[0][3] if len(points[0]) > 3 else 0
        stroke_points = [(current_x, -current_y, current_t)] 
        
        true_list.append(true[j])

        i = 2
        if len(points) > 1:
            vx = points[1][0]
            vy = points[1][1]
            vt = points[1][2] if len(points[1]) > 2 else 0
            vf = points[1][3] if len(points[1]) > 3 else 0

            current_x += vx
            current_y += vy
            current_t += vt
            current_f += vf
            stroke_points.append((current_x, -current_y, current_t))

            while i < len(points):
                vx += points[i][0]
                vy += points[i][1]
                vt += points[i][2] if len(points[i]) > 2 else 0
                vf += points[i][3] if len(points[i]) > 3 else 0
                
                current_x += vx
                current_y += vy
                current_t += vt
                current_f += vf
                stroke_points.append((current_x, -current_y, current_t))
                i += 1
                
        strokes_list.append(stroke_points)

    shift_x = 0
    shift_y = 0
    for stroke in strokes_list:
        for point in stroke:
            if shift_x > point[0]: shift_x = point[0]
            if shift_y > point[1]: shift_y = point[1]
    
    shifted_strokes_list = []
    for stroke in strokes_list:
        shifted_stroke = []
        for point in stroke:
            shifted_stroke.append((point[0] - shift_x, point[1] - shift_y, point[2]))
        shifted_strokes_list.append(shifted_stroke)

    return shifted_strokes_list, true_list

def extract_raw_strokes(root, canvas_matrix):
    """Отримує всі сирі штрихи з файлу і застосовує матрицю Canvas"""
    traces = root.findall('.//{http://www.w3.org/2003/InkML}trace')
    if not traces:
        traces = root.findall('.//trace')
    
    strokes = {}
    for trace in traces:
        if trace.text is None: continue
        tmp = []
        for sub_trace in trace.text.strip().split(','):
            clean_text = sub_trace.replace("'", " ").replace('"', " ").replace('-', " -").strip()
            parts = clean_text.split(' ')
            tmp.append([float(p) for p in parts if p])
        
        if canvas_matrix is not None:
            tmp = apply_canvas_transform(tmp, canvas_matrix)
            
        trace_id = trace.attrib.get(list(trace.attrib.keys())[0])
        if trace_id:
            strokes[trace_id] = tmp
    return strokes

def parse_inkml(file_path: str):
    tree = ET.parse(file_path)
    root = tree.getroot()
    canvas_matrix = get_canvas_transform(root)
    
    strokes = extract_raw_strokes(root, canvas_matrix)
    if not strokes:
        raise ValueError("No traces found in the InkML file.")
        
    true_y = get_labeled_strokes(root)
    strokes_list, true_y_list = to_absolute_coords_and_y_align(strokes, true_y)
    return strokes_list, true_y_list

def get_complex_cases(file_path: str):
    """Парсить файл і повертає лише ті фрагменти (ієрархії), де текст і графіка поруч"""
    tree = ET.parse(file_path)
    root = tree.getroot()
    canvas_matrix = get_canvas_transform(root)
    
    raw_strokes = extract_raw_strokes(root, canvas_matrix)
    true_y = get_labeled_strokes(root)
    
    exact_types = {}
    def _walk_exact(node, current_label):
        new_label = current_label
        for child in node:
            if child.tag.endswith('annotation') and child.attrib.get('type') == 'type':
                new_label = child.text
        
        ref = node.attrib.get('traceDataRef')
        if ref:
            exact_types[ref[1:]] = new_label
            return

        for child in node:
            if child.tag.endswith('traceView'):
                _walk_exact(child, new_label)

    for child in root:
        if child.tag.endswith('traceView'):
            _walk_exact(child, None)

    TEXT_TYPES = {'Textblock', 'Textline', 'Word', 'List', 'Symbol'}
    GRAPHIC_TYPES = {'Drawing', 'Diagram', 'Table'}

    MARKING_TYPES = {
        'Marking', 'Marking_Encircling', 'Marking_Underline', 
        'Marking_Sideline', 'Marking_Bracket', 'Marking_Angle', 
        'Marking_Connection'
    }
    NONTEXT_TYPES = {
        'Drawing', 'Diagram', 'Arrow', 'Formula', 'Symbol',
        'Table', 'Structure', 'Marking', 'Marking_Encircling',
        'Correction', 'Marking_Underline', 'Marking_Sideline', 'Marking_Bracket', 
        'Marking_Angle', 'Marking_Connection', 'Document', 'Garbage'
    }

    def parse_node(node):
        info = {
            'id': node.attrib.get('id', 'unknown'),
            'trace_refs': [],
            'children': [],
            'has_text': False,
            'has_graphic': False
        }
        
        for child in node:
            if child.tag.endswith('annotation') and child.attrib.get('type') == 'type':
                ann_text = child.text
                if ann_text in TEXT_TYPES: info['has_text'] = True
                if ann_text in GRAPHIC_TYPES: info['has_graphic'] = True
        
        for child in node:
            if child.tag.endswith('traceView'):
                ref = child.attrib.get('traceDataRef')
                if ref:
                    trace_id = ref[1:]
                    info['trace_refs'].append(trace_id)
                    # Перевіряємо заздалегідь згенеровані мітки штрихів
                    if trace_id in true_y:
                        if true_y[trace_id] == 1: info['has_text'] = True
                        if true_y[trace_id] == 0: info['has_graphic'] = True
                        
                else:
                    child_info = parse_node(child)
                    info['children'].append(child_info)
                    if child_info['has_text']: info['has_text'] = True
                    if child_info['has_graphic']: info['has_graphic'] = True
        return info

    def get_all_traces(info):
        t = list(info['trace_refs'])
        for c in info['children']:
            t.extend(get_all_traces(c))
        return t

    trees = []
    for child in root:
        if child.tag.endswith('traceView') and 'traceDataRef' not in child.attrib:
            trees.append(parse_node(child))

    complex_cases = []
    def find_lowest_mixed(info):
        if info['has_text'] and info['has_graphic']:
            child_has_both = any(c['has_text'] and c['has_graphic'] for c in info['children'])
            if not child_has_both:
                traces_in_node = get_all_traces(info)

                nontext_count = 0
                marking_count = 0

                for tid in traces_in_node:
                    t_type = exact_types.get(tid)
                    if t_type in NONTEXT_TYPES:
                        nontext_count += 1
                        if t_type in MARKING_TYPES:
                            marking_count += 1
                    
                if nontext_count > 0:
                    marking_ratio = marking_count / nontext_count
                    if marking_ratio > 0.5:
                        return # Ігноруємо цей випадок (більше 80% не-тексту є позначками)

                group_strokes = {tid: raw_strokes[tid] for tid in traces_in_node if tid in raw_strokes}
                group_labels = {tid: true_y[tid] for tid in traces_in_node if tid in true_y}
                
                if group_strokes:
                    aligned_strokes, aligned_labels = to_absolute_coords_and_y_align(group_strokes, group_labels)
                    complex_cases.append((aligned_strokes, aligned_labels, info['id']))
                return 
        
        for c in info['children']:
            find_lowest_mixed(c)

    for t in trees:
        find_lowest_mixed(t)

    return complex_cases


def save_batch(path, strokes_batch, labels_batch):
    torch.save({'strokes': strokes_batch, 'labels': labels_batch}, path)

def rotate(points, angle_deg):
    if not points:
        return points

    is_single = False
    first = points[0]
    if isinstance(first, (int, float)) or (isinstance(first, (list, tuple)) and isinstance(first[0], (int, float))):
        strokes = [points]
        is_single = True
    else:
        strokes = points

    xs, ys = [], []
    for stroke in strokes:
        for p in stroke:
            xs.append(p[0])
            ys.append(p[1])

    if not xs: return points

    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)

    angle = -math.radians(angle_deg) if angle_deg >= 0 else math.radians(-angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    rotated_strokes = []
    for stroke in strokes:
        new_stroke = []
        for p in stroke:
            x, y = p[0], p[1]
            dx = x - cx
            dy = y - cy
            rx = dx * cos_a - dy * sin_a + cx
            ry = dx * sin_a + dy * cos_a + cy
            if len(p) > 2:
                new_stroke.append((rx, ry, *p[2:]))
            else:
                new_stroke.append((rx, ry))
        rotated_strokes.append(new_stroke)

    min_x = min(p[0] for s in rotated_strokes for p in s)
    min_y = min(p[1] for s in rotated_strokes for p in s)
    shift_x = -min_x if min_x < 0 else 0.0
    shift_y = -min_y if min_y < 0 else 0.0

    if shift_x != 0.0 or shift_y != 0.0:
        shifted = []
        for s in rotated_strokes:
            ns = []
            for p in s:
                if len(p) > 2:
                    ns.append((p[0] + shift_x, p[1] + shift_y, *p[2:]))
                else:
                    ns.append((p[0] + shift_x, p[1] + shift_y))
            shifted.append(ns)
        rotated_strokes = shifted

    return rotated_strokes[0] if is_single else rotated_strokes

def scale(points, scale_factor_x, scale_factor_y):
    scaled_strokes = []
    for stroke in points:
        new_stroke = []
        for p in stroke:
            new_stroke.append((p[0] * scale_factor_x, p[1] * scale_factor_y, p[2]))
        scaled_strokes.append(new_stroke)
    return scaled_strokes


# --- ОСНОВНИЙ ЦИКЛ ---
if __name__ == '__main__':
    makedirs('batches', exist_ok=True)

    onlyfiles = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.inkml'))]
    aug_times=2
    
    for file in onlyfiles:
        file_path = join(mypath, file)
        print(f"Обробка {file}...")
        
        try:
            complex_cases = get_complex_cases(file_path)
            for idx, (c_strokes, c_labels, group_id) in enumerate(complex_cases):
                if c_strokes.__len__() < 5: continue
                save_name = f'batches/sepbatch{file[:-6]}_{idx}.pt'
                save_batch(save_name, [c_strokes], [c_labels])
                if c_labels.__len__()==0:
                    print(f"Warning: No labels found for complex case {group_id} in file {file}")

                for i in range(aug_times):
                    strokes_rotated = rotate(c_strokes, random.uniform(-15.0, 15.0))
                    strokes_scaled = scale(strokes_rotated, random.uniform(0.8, 1.2), random.uniform(0.8, 1.2))
                    strokes_batch_aug = [strokes_scaled]
                    save_batch(f'batches/sepaug{i}_batch{file[:-6]}_{idx}.pt', strokes_batch_aug, [c_labels])
            
            strokes, true_y = parse_inkml(file_path)
            strokes_batch = [strokes]
            true_y_batch = [true_y]
            save_batch(f'batches/batch{file[:-6]}.pt', strokes_batch, true_y_batch)

            for i in range(aug_times):
                strokes_rotated = rotate(strokes, random.uniform(-15.0, 15.0))
                strokes_scaled = scale(strokes_rotated, random.uniform(0.8, 1.2), random.uniform(0.8, 1.2))
                strokes_batch_aug = [strokes_scaled]
                save_batch(f'batches/aug{i}_batch{file[:-6]}.pt', strokes_batch_aug, true_y_batch)
                
        except Exception as e:
            print(f"Помилка при обробці {file}: {e}")

    print(f"Знайдені унікальні типи: {uniqe_types}")
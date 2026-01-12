from os import listdir
from os.path import isfile, join
import random
import xml.etree.ElementTree as ET
import torch


mypath = './IAMonDo-db-1.0/'
batch=1
uniqe_types = []
def parse_inkml(file_path: str):

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
            'Garbage'                                       # Сміття
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
            # scrap f to save memory
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

        return strokes_list, true_list

    tree = ET.parse(file_path)

    root = tree.getroot()

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
        strokes[trace.attrib[list(trace.attrib.keys())[0]]]=tmp
    
    true_y=get_labeled_strokes(root)
    
    strokes, true_y=to_absolute_coords_and_y_align(strokes, true_y)

    return (strokes, true_y)

def save_batch(path, strokes_batch, labels_batch):
    torch.save({'strokes': strokes_batch, 'labels': labels_batch}, path)

onlyfiles = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and f.endswith('.inkml'))]

random.shuffle(onlyfiles)
batches = [onlyfiles[i:i + batch] for i in range(0, len(onlyfiles), batch)]

i=0
for batch_files in batches:
    strokes_batch = []
    true_y_batch = []
    for file in batch_files:
        file_path = join(mypath, file)
        strokes, true_y = parse_inkml(file_path)

        strokes_batch.append(strokes)
        true_y_batch.append(true_y)
    save_batch(f'batches/batch{i}.pt', strokes_batch, true_y_batch)
    i+=1

print(f'Total batches created: {i}')
print(uniqe_types)
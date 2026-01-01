from os import listdir
from os.path import isfile, join
import random
from typing import List
import xml.etree.ElementTree as ET
import numpy as np
import torch


mypath = './IAMonDo-db-1.0/'
batch=16
uniqe_types = []
def parse_inkml(file_path: str) -> tuple[dict, dict]:

    def get_labeled_strokes(root, num_strokes) -> dict:
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
            'Garbage'                                       # Сміття
        }
        def _walk(node, current_label):
            """Внутрішня функція для обходу дерева зі збереженням контексту"""
            new_label = current_label
             
            # 1. Шукаємо анотацію типу у поточному вузлі (через ітерацію, щоб ігнорувати namespaces)
            for child in node:
                if child.tag.endswith('annotation') and child.attrib.get('type') == 'type':
                    ann_text = child.text
                    if ann_text not in uniqe_types:
                        uniqe_types.append(ann_text)
                    if ann_text in TEXT_TYPES:
                        new_label = 1 # Клас: Текст
                    elif ann_text in NONTEXT_TYPES:
                        new_label = 0 # Клас: Не текст
                    else:
                        new_label = -1 # Клас: Невідомий
        
            # 2. Якщо це штрих (є посилання traceDataRef), записуємо результат
            ref = node.attrib.get('traceDataRef')
            if ref:
                
                dataset[ref[1:]]=new_label
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
    
    true_y=get_labeled_strokes(root, strokes.__len__())
    
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
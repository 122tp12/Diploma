import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy
from typing import List
import features
"""
#for with force-based linewidths
from matplotlib.collections import LineCollection
import numpy as np
import re
"""

def parse_inkml_and_plot(file_path: str):
    try:
        tree = ET.parse(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return
    except ET.ParseError:
        print(f"Error parsing XML. Check if the file has valid tags.")
        return

    root = tree.getroot()

    traces = root.findall('.//{http://www.w3.org/2003/InkML}trace')
    if not traces:
        traces = root.findall('.//trace')

    if not traces:
        print("No traces found.")
        return
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

    
        """
        With force-based linewidths
        fs = [abs(f)/50 for f in fs]
        
        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        lc = LineCollection(segments, linewidths=fs[:-1], color='black')
        plt.gca().add_collection(lc)
        """
    
    feat=features.extract_stroke_features(strokes)

    plt.axis('equal')
    plt.gca().axis('off')
    plt.title("Render InkML")
    plt.show()

parse_inkml_and_plot('./IAMonDo-db-1.0/001.inkml')
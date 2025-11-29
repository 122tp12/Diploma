import xml.etree.ElementTree as ET
import re
import matplotlib.pyplot as plt

def parse_inkml_and_plot(file_path):
    # 1. Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Define namespaces (InkML usually uses this namespace)
    namespaces = {'inkml': 'http://www.w3.org/2003/InkML'}

    # 2. Find all <trace> tags
    # Try finding with namespace first, then fallback to without
    traces = root.findall('.//{http://www.w3.org/2003/InkML}trace')
    if not traces:
        traces = root.findall('.//trace')

    if not traces:
        print("No traces found.")
        return

    all_strokes = []

    # 3. Iterate through each trace (stroke)
    for trace in traces:
        trace_string = trace.text.strip()
        
        # Regex to parse numbers, handling mashed values like ".001-4"
        # It looks for optional negatives, followed by digits and optional decimals
        pattern = r'-?(?:\d+(?:\.\d*)?|\.\d+)'
        values = [float(x) for x in re.findall(pattern, trace_string)]

        # We need at least one point (4 channels: X, Y, T, F)
        if len(values) < 4:
            continue

        stroke_x = []
        stroke_y = []

        # --- Point 1: Absolute Coordinates ---
        # The first group of 4 numbers are absolute values
        curr_x = values[0]
        curr_y = values[1]
        
        stroke_x.append(curr_x)
        stroke_y.append(curr_y)

        # --- Subsequent Points: Delta Values ---
        # Loop through the rest of the data in chunks of 4
        # Channels order is defined in the file as: X, Y, T, F
        for i in range(4, len(values), 4):
            # Ensure we have a full set of 4 values
            if i + 3 < len(values):
                # Extract deltas
                dx = values[i]
                dy = values[i+1]
                dt = values[i+2] # Time delta (not used for plotting)
                # df = values[i+3] # Force delta (not used for plotting)

                # Update absolute position (Previous + Delta)
                curr_x += dx
                curr_y += dy
                
                stroke_x.append(curr_x)
                stroke_y.append(curr_y)
        
        all_strokes.append((stroke_x, stroke_y))

    # 4. Plot the reconstructed strokes
    plt.figure(figsize=(10, 8))
    for sx, sy in all_strokes:
        # Plot (x, -y) because screen coordinates usually have Y going down,
        # but matplotlib has Y going up. Inverting Y orients it correctly.
        plt.plot(sx, [-y for y in sy], color='black', linewidth=1)
    
    plt.axis('equal')
    plt.title("Handwriting from 001.inkml")
    plt.gca().axis('off') # Hide axes for a clean look
    plt.show()
    return all_strokes
def get_traces_data(inkml_file_abs_path, xmlns='{http://www.w3.org/2003/InkML}'):

    traces_data = []

    tree = ET.parse(inkml_file_abs_path)
    root = tree.getroot()
    # doc_namespace = "{http://www.w3.org/2003/InkML}"
    doc_namespace = xmlns

    'Stores traces_all with their corresponding id'
    traces_all = [{'id': trace_tag.get('id'),
                   'coords': [[round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000)
                               for axis_coord in coord[1:].split(' ')] if coord.startswith(' ')
                              else [round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000)
                                    for axis_coord in coord.split(' ')]
                              for coord in (trace_tag.text).replace('\n', '').split(',')]}
                  for trace_tag in root.findall(doc_namespace + 'trace')]

    'Sort traces_all list by id to make searching for references faster'
    traces_all.sort(key=lambda trace_dict: int(trace_dict['id']))

    'Always 1st traceGroup is a redundant wrapper'
    traceGroupWrapper = root.find(doc_namespace + 'traceGroup')

    if traceGroupWrapper is not None:
        for traceGroup in traceGroupWrapper.findall(doc_namespace + 'traceGroup'):

            label = traceGroup.find(doc_namespace + 'annotation').text

            'traces of the current traceGroup'
            traces_curr = []
            for traceView in traceGroup.findall(doc_namespace + 'traceView'):

                'Id reference to specific trace tag corresponding to currently considered label'
                traceDataRef = int(traceView.get('traceDataRef'))

                'Each trace is represented by a list of coordinates to connect'
                single_trace = traces_all[traceDataRef]['coords']
                traces_curr.append(single_trace)

            traces_data.append({'label': label, 'trace_group': traces_curr})

    else:
        'Consider Validation data that has no labels'
        [traces_data.append({'trace_group': [trace['coords']]})
         for trace in traces_all]

    return traces_data

traces2=get_traces_data('./IAMonDo-db-1.0/001.inkml')
traces=parse_inkml_and_plot('./IAMonDo-db-1.0/001.inkml')
print(traces2)
import json
import numpy as np
import torch
from scipy.spatial import ConvexHull
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist

type list_strokes = list[list[list[float]]]

# Utility functions to create edge by proximity and time
def get_edges(strokes: list_strokes, proxy_threshold: float = 50.0, time_threshold: float=2.0) -> list[tuple[int, int]]:
    edges = []

    for i in range(len(strokes)):
        for j in range(len(strokes)):
            if i == j:
                continue
            if np.abs(strokes[i][-1][2]-strokes[j][0][2])<time_threshold or np.abs(strokes[i][0][2]-strokes[j][-1][2])<time_threshold:
                edges.append((i, j))
                continue
            for point1 in strokes[i]:
                for point2 in strokes[j]:
                    distance = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
                    if distance < proxy_threshold:
                        edges.append((i, j))
                        break

    return edges

# Node features:
# 12, 13
def get_normalization_features(strokes: list_strokes) -> dict:
    widths = []
    heights = []
    
    for s in strokes:
        points = np.array(s)
            
        w = np.max(points[:, 0]) - np.min(points[:, 0])
        h = np.max(points[:, 1]) - np.min(points[:, 1])
        
        widths.append(w)
        heights.append(h)

    median_h = np.median(heights)
    if median_h == 0: 
        median_h = 1.0 

    norm_widths = [w / median_h for w in widths]
    norm_heights = [h / median_h for h in heights]

    return {
        "normalized_width": norm_widths,
        "normalized_height": norm_heights
    }

# 8
def linearity_ratio(strokes: list_strokes) -> list[float]:
    scores = []
    for s in strokes:
        points = np.array(s)
        
        # Need at least 2 points to define a line
        if len(points) < 2:
            scores.append(1.0)
            continue
            
        start_point = points[0]
        end_point = points[-1]
        dist_end_to_end = np.linalg.norm(end_point - start_point)
        
        diffs = points[1:] - points[:-1]
        segment_lengths = np.linalg.norm(diffs, axis=1)
        total_trajectory_len = np.sum(segment_lengths)
        
        if total_trajectory_len == 0:
            scores.append(1.0)
        else:
            scores.append(dist_end_to_end / total_trajectory_len)
            
    return scores

# 9
def accumulated_curvature(strokes: list_strokes) -> list[float]:
    scores = []
    for s in strokes:
        points = np.array(s)
        
        # Need at least 3 points to form an angle
        if len(points) < 3:
            scores.append(0.0)
            continue
            
        diffs = points[1:] - points[:-1]
        
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        
        angle_changes = angles[1:] - angles[:-1]

        angle_changes = np.mod(angle_changes + np.pi, 2 * np.pi) - np.pi
        
        total_curvature = np.sum(np.abs(angle_changes))
        scores.append(total_curvature)
        
    return scores

# 1
def trajectory_length(strokes: list_strokes) -> list[float]:
    lengths = []
    for s in strokes:
        points = np.array(s)
        if len(points) < 2:
            lengths.append(0.0)
            continue
            
        diffs = points[1:] - points[:-1]
        segment_lengths = np.linalg.norm(diffs, axis=1)
        lengths.append(float(np.sum(segment_lengths)))
        
    return lengths

# 2
def area_of_convex_hull(strokes: list_strokes) -> list[float]:

    areas = []
    for s in strokes:
        points = np.array(s)

        # Convex Hull needs at least 3 points.
        if len(points) < 3:
            areas.append(0.0)
            continue
            
        try:
            hull = ConvexHull(points)
            areas.append(float(hull.volume))
        except Exception:
            areas.append(0.0)
            
    return areas

# 3 
def trajectory_duration(strokes: list_strokes) -> list[float]:

    durations = []
    for s in strokes:
        #If data is [x, y, t], calc t_end - t_start
        if len(s) > 0 and len(s[0]) >= 3:
             t_start = s[0][2]
             t_end = s[-1][2]
             durations.append(float(t_end - t_start))
             
        #If data is only [x, y], use point count as proxy for time
        else:
             durations.append(float(len(s)))
             
    return durations

# 15
def number_of_spatial_neighbors(strokes: list_strokes, threshold: float = 50.0) -> list[float]:
    if not strokes:
        return []

    centroids = []
    for s in strokes:
        points = np.array(s)
        if len(points) > 0:
            center = np.mean(points[:, :2], axis=0)
            centroids.append(center)
        else:
            centroids.append([0, 0])
    
    centroids = np.array(centroids)

    tree = KDTree(centroids)
    indices_list = tree.query_radius(centroids, r=threshold)
    counts = [float(len(ind) - 1) for ind in indices_list]
    
    return counts

# 14
def number_of_temporal_neighbors(strokes: list_strokes, time_threshold: float = 2.0) -> list[float]:
    counts = []

    has_time = (len(strokes) > 0 and len(strokes[0]) > 0 and len(strokes[0][0]) >= 3)
    
    if not has_time:
        total = len(strokes)
        for i in range(total):
            neighbors = 0
            if i > 0: neighbors += 1
            if i < total - 1: neighbors += 1
            counts.append(float(neighbors))
        return counts

    intervals = []
    for s in strokes:
        intervals.append((s[0][2], s[-1][2])) # (start_time, end_time)
        
    for i in range(len(strokes)):
        current_start, current_end = intervals[i]
        count = 0
        for j in range(len(strokes)):
            if i == j: continue
            
            other_start, other_end = intervals[j]
            

            if other_start > current_end:
                gap = other_start - current_end
            elif current_start > other_end:
                gap = current_start - other_end
            else:
                gap = 0
                
            if gap < time_threshold:
                count += 1
        counts.append(float(count))
        
    return counts

# 6
def circular_variance(strokes: list) -> list[float]:
    scores = []
    for s in strokes:
        points = np.array(s)
        if len(points) < 2:
            scores.append(0.0)
            continue
            
        centroid = np.mean(points[:, :2], axis=0)
        
        radii = np.linalg.norm(points[:, :2] - centroid, axis=1)

        mean_radius = np.mean(radii)
        if mean_radius == 0:
            scores.append(0.0)
        else:
            var_radius = np.var(radii)
            scores.append(var_radius / (mean_radius ** 2))
            
    return scores

# 7
def principal_axis_features(strokes: list) -> list[float]:
    scores = []
    for s in strokes:
        points = np.array(s)[:, :2]
        if len(points) < 3:
            scores.append(0.0)
            continue
            
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        
        cov = np.cov(centered_points, rowvar=False)
        
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        

        principal_axis = eig_vecs[:, -1]
        
        projections = np.dot(points, principal_axis)
        
        proj_mean = np.mean(projections)
        proj_min = np.min(projections)
        proj_max = np.max(projections)
        
        proj_midpoint = (proj_max + proj_min) / 2
        
        stroke_length_on_axis = proj_max - proj_min
        
        if stroke_length_on_axis == 0:
            scores.append(0.0)
        else:
            offset = proj_mean - proj_midpoint
            scores.append(offset / stroke_length_on_axis)
            
    return scores

# 10, 11
def perpendicularity_features(strokes: list) -> dict:
    squared_scores = []
    signed_scores = []
    
    for s in strokes:
        points = np.array(s)[:, :2]
        if len(points) < 3:
            squared_scores.append(0.0)
            signed_scores.append(0.0)
            continue
            
        start_p = points[0]
        end_p = points[-1]
        
        line_vec = end_p - start_p
        line_len_sq = np.dot(line_vec, line_vec)
        
        if line_len_sq == 0:
            dists = np.linalg.norm(points - start_p, axis=1)
            squared_scores.append(np.sum(dists**2))
            signed_scores.append(0.0) 
            continue
        
        vec_from_start = points - start_p

        cross_prods = vec_from_start[:, 0] * line_vec[1] - vec_from_start[:, 1] * line_vec[0]

        line_len = np.sqrt(line_len_sq)
        perp_distances = cross_prods / line_len
        

        sq_perp = np.sum(perp_distances ** 2)
        squared_scores.append(sq_perp)

        signed_perp = np.sum(perp_distances)
        signed_scores.append(signed_perp)
        
    return {
        "squared_perpendicularity": squared_scores,
        "signed_perpendicularity": signed_scores
    }

# Edge features:
# 5, 6
def centroid_distance_features(strokes: list_strokes, edges: list[tuple[int, int]]) -> dict:
    centroids = []
    for s in strokes:
        points = np.array(s)
        if len(points) > 0:
            center = np.mean(points[:, :2], axis=0)
            centroids.append(center)
        else:
            centroids.append(np.array([0.0, 0.0]))
            
    horiz_dists = []
    vert_dists = []
    
    for u, v in edges:
        c1 = centroids[u]
        c2 = centroids[v]
        
        h_dist = abs(c1[0] - c2[0])
        horiz_dists.append(float(h_dist))
        
        v_dist = abs(c1[1] - c2[1])
        vert_dists.append(float(v_dist))
        
    return {
        "horizontal_dist": horiz_dists,
        "vertical_dist": vert_dists
    }

# 1
def calculate_min_distance_edge(strokes: list_strokes, edges: list[tuple[int, int]]) -> list[float]:
    min_dists = []
    
    for u, v in edges:
        pts_u = np.array(strokes[u])[:, :2]
        pts_v = np.array(strokes[v])[:, :2]
        
        if len(pts_u) == 0 or len(pts_v) == 0:
            min_dists.append(0.0)
            continue

        dists = cdist(pts_u, pts_v)
        min_dists.append(float(np.min(dists)))
        
    return min_dists

# 12, 13, 14, 16
def calculate_bbox_ratios(strokes: list_strokes, edges: list[tuple[int, int]]) -> dict:
    bboxes = []
    
    for s in strokes:
        pts = np.array(s)
        if len(pts) == 0:
            bboxes.append(None)
            continue
            
        min_x, min_y = np.min(pts[:, :2], axis=0)
        max_x, max_y = np.max(pts[:, :2], axis=0)
        
        w = max_x - min_x
        h = max_y - min_y
        area = w * h
        
        bboxes.append({
            'w': w, 'h': h, 'area': area,
            'x1': min_x, 'y1': min_y, 'x2': max_x, 'y2': max_y
        })
        
    feat_12 = [] # Ratio of area of largest BB to their Union
    feat_13 = [] # Ratio of Widths
    feat_14 = [] # Ratio of Heights
    feat_16 = [] # Ratio of Areas
    
    for u, v in edges:
        b1 = bboxes[u]
        b2 = bboxes[v]
        
        if b1 is None or b2 is None:
            feat_12.append(0.0); feat_13.append(0.0)
            feat_14.append(0.0); feat_16.append(0.0)
            continue
        
        w_ratio = min(b1['w'], b2['w']) / max(b1['w'], b2['w'], 1e-6)
        h_ratio = min(b1['h'], b2['h']) / max(b1['h'], b2['h'], 1e-6)
        area_ratio = min(b1['area'], b2['area']) / max(b1['area'], b2['area'], 1e-6)
        
        feat_13.append(float(w_ratio))
        feat_14.append(float(h_ratio))
        feat_16.append(float(area_ratio))
        
        dx = min(b1['x2'], b2['x2']) - max(b1['x1'], b2['x1'])
        dy = min(b1['y2'], b2['y2']) - max(b1['y1'], b2['y1'])
        
        if (dx >= 0) and (dy >= 0):
            intersection_area = dx * dy
        else:
            intersection_area = 0.0
            
        union_area = b1['area'] + b2['area'] - intersection_area
        
        largest_individual_area = max(b1['area'], b2['area'])
        
        if union_area == 0:
            feat_12.append(0.0)
        else:
            feat_12.append(largest_individual_area / union_area)
            
    return {
        "ratio_width": feat_13,
        "ratio_height": feat_14,
        "ratio_area": feat_16,
        "ratio_largest_to_union": feat_12
    }

# 9, 10, 11
def temporal_edge_features(strokes: list_strokes, edges: list[tuple[int, int]]) -> dict:
    stroke_info = []
    has_time = (len(strokes) > 0 and len(strokes[0]) > 0 and len(strokes[0][0]) >= 3)

    for i, s in enumerate(strokes):
        pts = np.array(s)
        if len(pts) == 0:
            stroke_info.append(None)
            continue
            
        start_pt = pts[0]
        end_pt = pts[-1]
        
        if has_time:
            t_start = start_pt[2]
            t_end = end_pt[2]
        else:
            t_start = float(i)
            t_end = float(i) + 1.0
            
        stroke_info.append({
            "p_start": start_pt[:2],
            "p_end": end_pt[:2], 
            "t_start": t_start,
            "t_end": t_end
        })
        
    feat_9 = []
    feat_10 = []
    feat_11 = []
    
    for u, v in edges:
        s1 = stroke_info[u]
        s2 = stroke_info[v]
        
        if s1 is None or s2 is None:
            feat_9.append(0.0)
            feat_10.append(0.0)
            feat_11.append(0.0)
            continue
        
        if s1['t_end'] <= s2['t_start']:
            t_dist = s2['t_start'] - s1['t_end']
            p_end_u = s1['p_end']
            p_start_v = s2['p_start']
        elif s2['t_end'] <= s1['t_start']:
            t_dist = s1['t_start'] - s2['t_end']
            p_end_u = s2['p_end']
            p_start_v = s1['p_start']
        else:
            t_dist = 0.0
            p_end_u = s1['p_end']
            p_start_v = s2['p_start']
        
        if t_dist < 1e-4: t_dist = 1e-4
        feat_9.append(float(t_dist))
        
        # Distances
        dx = abs(p_start_v[0] - p_end_u[0])
        dy = abs(p_start_v[1] - p_end_u[1])
        
        # Feature 10: Euclidean Speed (Direct line speed)
        euclidean_dist = np.sqrt(dx**2 + dy**2)
        feat_10.append(float(euclidean_dist / t_dist))
        
        # Feature 11: Manhattan Speed (Sum of X and Y speeds)
        # This combines both axes into one scalar
        manhattan_dist = dx + dy
        feat_11.append(float(manhattan_dist / t_dist))
        
    return {
        "temporal_dist": feat_9,
        "ratio_off_temporal": feat_10,
        "ratio_off_xy_temporal": feat_11 
    }

# 19
def ratio_of_curvatures(strokes: list_strokes, edges: list[tuple[int, int]]) -> list[float]:
    curvatures = accumulated_curvature(strokes)
    
    ratios = []
    for u, v in edges:
        c1 = curvatures[u]
        c2 = curvatures[v]
        
        # Avoid division by zero
        denom = max(c1, c2, 1e-6)
        num = min(c1, c2)
        
        ratios.append(float(num / denom))
        
    return ratios

def extract_stroke_features(strokes: list_strokes, offset, proxy_threshold, time_threshold) -> dict:
    edges = get_edges(strokes, proxy_threshold, time_threshold)

    norm_feats = get_normalization_features(strokes)
    perpen_feats = perpendicularity_features(strokes)
    centroid_dists = centroid_distance_features(strokes, edges)
    bbox_ratios = calculate_bbox_ratios(strokes, edges)
    temp_edge_feats = temporal_edge_features(strokes, edges)
    nodes_out = {
            "normalized_width": norm_feats["normalized_width"],   # 12
            "normalized_height": norm_feats["normalized_height"],     # 13
            "linearity_ratio": linearity_ratio(strokes),       # 8
            "accumulated curvature": accumulated_curvature(strokes), # 9

            "num_temporal_neighbours": number_of_temporal_neighbors(strokes, time_threshold), # 14
            "num_spatioal_neighbours": number_of_spatial_neighbors(strokes, proxy_threshold), # 15
            "trajectory_length": trajectory_length(strokes), # 1
            "Trajectory_duration": trajectory_duration(strokes), # 3
            "area_convex_hull": area_of_convex_hull(strokes), # 2

            "accumulated_squared_perpendicularity": perpen_feats["squared_perpendicularity"], # 10
            "accumulated_signed_perpendicularity": perpen_feats["signed_perpendicularity"], # 11
            "circular_variance": circular_variance(strokes), # 6
            "normalized_offset_along_pricipal_axis": principal_axis_features(strokes), # 7
        }

    edges_out = {
            "vertical_distance_between_centroids": centroid_dists["vertical_dist"],   # 6
            "horizontal_distance_between_centroids": centroid_dists["horizontal_dist"], # 5
            "minimum_distance": calculate_min_distance_edge(strokes, edges), #1

            "ratio_largest_to_union": bbox_ratios["ratio_largest_to_union"], # 12
            "ratio_width": bbox_ratios["ratio_width"],                    # 13
            "ratio_height": bbox_ratios["ratio_height"],                 # 14
            "ratio_area": bbox_ratios["ratio_area"],                     # 16

            "temporal_dist": temp_edge_feats["temporal_dist"],                                  # 9
            "ratio_off_temporal": temp_edge_feats["ratio_off_temporal"],      # 10
            "ratio_off_xy_temporal": temp_edge_feats["ratio_off_xy_temporal"], # 11
            "ratio_curvatures": ratio_of_curvatures(strokes, edges),                                                     # 19
        }
    
    applyed_offset_edges = [(a+offset, b+offset) for (a,b) in edges]


    return {
        "nodes": nodes_out,
        "edge_index": applyed_offset_edges,
        "edges_features": edges_out
    }

def get_masks(num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    perm = torch.randperm(num_nodes)

    separ = int(0.85 * num_nodes)

    train_idx = perm[:separ]
    val_idx = perm[separ:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True

    return (train_mask, val_mask)
import json
import numpy as np
import torch
from scipy.spatial import ConvexHull
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist

type list_strokes = list[list[list[float]]]

# Utility functions to create edge by proximity and time
def get_edges(strokes: list_strokes, proxy_threshold: float = 20.0, time_threshold: float=2.0) -> list[tuple[int, int]]:
    edges = []
    for i in range(len(strokes)):
        for j in range(i + 1, len(strokes)):
            
            if np.abs(strokes[i][-1][2]-strokes[j][0][2]) < time_threshold or \
               np.abs(strokes[i][0][2]-strokes[j][-1][2]) < time_threshold:
                edges.append((i, j))
                edges.append((j, i))
                continue
            for point1 in strokes[i]:
                for point2 in strokes[j]:
                    distance = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
                    if distance < proxy_threshold:
                        edges.append((i, j))
                        edges.append((j, i))
                        break
                else:
                    continue  # only executed if the inner loop did NOT break
                break  # only executed if the inner loop DID break

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
        
        points_2d = points[:, :2]
        dist_end_to_end = np.linalg.norm(points_2d[-1] - points_2d[0])
        diffs = points_2d[1:] - points_2d[:-1]
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
            
        points_2d = points[:, :2]
        diffs = points_2d[1:] - points_2d[:-1]
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
            hull = ConvexHull(points[:, :2])
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

# 4
def principal_axis_ratio(strokes: list) -> list[float]:
    ratios = []
    for s in strokes:
        points = np.array(s)[:, :2]
        if len(points) < 3:
            ratios.append(0.0)
            continue
            
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        
        cov = np.cov(centered_points, rowvar=False)
        
        eig_vals = np.linalg.eigvalsh(cov)
        
        max_eig = np.max(eig_vals)
        min_eig = np.min(eig_vals)
        
        if max_eig <= 0:
            ratios.append(1.0)
        else:
            min_eig = max(0.0, min_eig)
            ratios.append(np.sqrt(min_eig / max_eig))
            
    return ratios

# 5
def rectangularity(strokes: list) -> list[float]:
    scores = []
    for s in strokes:
        points = np.array(s)[:, :2]
        
        if len(points) < 3:
            scores.append(1.0)
            continue
            
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
            
            if hull_area == 0:
                scores.append(1.0)
                continue

            hull_points = points[hull.vertices]
            min_bbox_area = float('inf')
            
            diffs = hull_points[1:] - hull_points[:-1]
            diffs = np.vstack([diffs, hull_points[0] - hull_points[-1]])
            
            for i, edge in enumerate(diffs):
                norm = np.linalg.norm(edge)
                if norm == 0: continue
                
                u = edge / norm
                v = np.array([-u[1], u[0]])

                proj_u = np.dot(hull_points, u)
                proj_v = np.dot(hull_points, v)
                
                w = np.max(proj_u) - np.min(proj_u)
                h = np.max(proj_v) - np.min(proj_v)
                
                area = w * h
                if area < min_bbox_area:
                    min_bbox_area = area

            if min_bbox_area <= 0:
                 scores.append(0.0)
            else:
                 scores.append(hull_area / min_bbox_area)

        except Exception:
            scores.append(0.0)
            
    return scores

# 16, 17, 18, 19
def get_time_neighbor_statistics(strokes: list, time_threshold: float = 2.0) -> dict:
    """
    Calculates statistics (Mean/Std) for Distances and Lengths of time neighbors.
    Time neighbors are defined as strokes separated by a time gap < time_threshold.
    
    Returns:
        dict: Keys for Features 16, 17, 18, 19
    """
    # 1. Pre-compute lengths (Feature 1 logic)
    # We duplicate the logic here or call the existing function if available globally
    # To ensure standalone correctness, I'll calculate lengths locally efficiently.
    lengths = []
    points_cache = []
    
    for s in strokes:
        pts = np.array(s)[:, :2]
        points_cache.append(pts)
        
        if len(pts) < 2:
            lengths.append(0.0)
        else:
            diffs = pts[1:] - pts[:-1]
            segment_lens = np.linalg.norm(diffs, axis=1)
            lengths.append(float(np.sum(segment_lens)))
            
    # 2. Determine Time Intervals
    intervals = []
    has_time = (len(strokes) > 0 and len(strokes[0]) > 0 and len(strokes[0][0]) >= 3)
    
    if has_time:
        for s in strokes:
            intervals.append((s[0][2], s[-1][2]))
    else:
        # Fallback: Use index as time proxy
        for i in range(len(strokes)):
            intervals.append((float(i), float(i) + 1.0))
            
    # 3. Calculate Stats
    feat_16 = [] # Avg Distance
    feat_17 = [] # Std Distance
    feat_18 = [] # Avg Length
    feat_19 = [] # Std Length
    
    n = len(strokes)
    
    for i in range(n):
        curr_start, curr_end = intervals[i]
        pts_i = points_cache[i]
        
        neighbor_dists = []
        neighbor_lens = []
        
        for j in range(n):
            if i == j: continue
            
            other_start, other_end = intervals[j]
            
            # Calculate gap
            if other_start > curr_end:
                gap = other_start - curr_end
            elif curr_start > other_end:
                gap = curr_start - other_end
            else:
                gap = 0.0
                
            if gap < time_threshold:
                # It is a time neighbor
                neighbor_lens.append(lengths[j])
                
                # Calculate Spatial Distance (Min Euclidean Distance)
                pts_j = points_cache[j]
                if len(pts_i) == 0 or len(pts_j) == 0:
                    dist = 0.0
                else:
                    # cdist returns matrix of all pair distances
                    dists = cdist(pts_i, pts_j)
                    dist = float(np.min(dists))
                neighbor_dists.append(dist)
                
        if not neighbor_dists:
            feat_16.append(0.0)
            feat_17.append(0.0)
            feat_18.append(0.0)
            feat_19.append(0.0)
        else:
            feat_16.append(float(np.mean(neighbor_dists)))
            feat_17.append(float(np.std(neighbor_dists)))
            feat_18.append(float(np.mean(neighbor_lens)))
            feat_19.append(float(np.std(neighbor_lens)))

    return {
        "avg_dist_time_neighbors": feat_16,  # 16
        "std_dist_time_neighbors": feat_17,  # 17
        "avg_len_time_neighbors": feat_18,   # 18
        "std_len_time_neighbors": feat_19    # 19
    }

# 20, 21, 22, 23
def get_spatial_neighbor_statistics(strokes: list, threshold: float = 50.0) -> dict:
    """
    Calculates statistics (Mean/Std) for Distances and Lengths of spatial neighbors.
    Spatial neighbors are identified using KDTree on centroids (consistent with Feature 15).
    
    Distances calculated are the Minimum Euclidean Distances between stroke points.
    """
    # 1. Pre-compute lengths and centroids
    lengths = []
    points_cache = []
    centroids = []
    
    for s in strokes:
        pts = np.array(s)[:, :2]
        points_cache.append(pts)
        
        # Length
        if len(pts) < 2:
            lengths.append(0.0)
        else:
            diffs = pts[1:] - pts[:-1]
            segment_lens = np.linalg.norm(diffs, axis=1)
            lengths.append(float(np.sum(segment_lens)))
            
        # Centroid
        if len(pts) > 0:
            center = np.mean(pts, axis=0)
            centroids.append(center)
        else:
            centroids.append([0.0, 0.0])
            
    centroids = np.array(centroids)
    
    # 2. Find Spatial Neighbors using KDTree
    # Note: KDTree requires at least one point
    if len(centroids) == 0:
         return {
            "avg_dist_space_neighbors": [], "std_dist_space_neighbors": [],
            "avg_len_space_neighbors": [], "std_len_space_neighbors": []
        }

    tree = KDTree(centroids)
    # query_radius returns an array of arrays (indices of neighbors)
    indices_list = tree.query_radius(centroids, r=threshold)
    
    feat_20 = [] # Avg Distance
    feat_21 = [] # Std Distance
    feat_22 = [] # Avg Length
    feat_23 = [] # Std Length
    
    for i, neighbors in enumerate(indices_list):
        pts_i = points_cache[i]
        
        neighbor_dists = []
        neighbor_lens = []
        
        for j in neighbors:
            if i == j: continue # Skip self
            
            # Store Length
            neighbor_lens.append(lengths[j])
            
            # Calculate Min Euclidean Distance between actual strokes
            pts_j = points_cache[j]
            if len(pts_i) == 0 or len(pts_j) == 0:
                dist = 0.0
            else:
                dists = cdist(pts_i, pts_j)
                dist = float(np.min(dists))
            neighbor_dists.append(dist)
            
        if not neighbor_dists:
            feat_20.append(0.0)
            feat_21.append(0.0)
            feat_22.append(0.0)
            feat_23.append(0.0)
        else:
            feat_20.append(float(np.mean(neighbor_dists)))
            feat_21.append(float(np.std(neighbor_dists)))
            feat_22.append(float(np.mean(neighbor_lens)))
            feat_23.append(float(np.std(neighbor_lens)))
            
    return {
        "avg_dist_space_neighbors": feat_20,
        "std_dist_space_neighbors": feat_21,
        "avg_len_space_neighbors": feat_22,
        "std_len_space_neighbors": feat_23
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

# 17, 18
def length_and_duration_ratios(strokes: list, edges: list[tuple[int, int]]) -> dict:
    """
    Calculates the ratio of lengths and durations between connected strokes.
    Ratio is defined as min(val1, val2) / max(val1, val2).
    """
    # Reuse existing node feature functions to get raw values
    lengths = trajectory_length(strokes)
    durations = trajectory_duration(strokes)
    
    ratio_len = []
    ratio_dur = []
    
    for u, v in edges:
        # Feature 17: Ratio of lengths
        l1 = lengths[u]
        l2 = lengths[v]
        # Avoid division by zero
        denom_l = max(l1, l2, 1e-6)
        ratio_len.append(min(l1, l2) / denom_l)
        
        # Feature 18: Ratio of durations
        d1 = durations[u]
        d2 = durations[v]
        # Avoid division by zero
        denom_d = max(d1, d2, 1e-6)
        ratio_dur.append(min(d1, d2) / denom_d)
        
    return {
        "ratio_length": ratio_len,     # 17
        "ratio_duration": ratio_dur    # 18
    }

# 7, 8
def off_stroke_features(strokes: list, edges: list[tuple[int, int]]) -> dict:
    """
    Calculates the spatial properties of the gap between two connected strokes.
    
    Determines the order of strokes (u->v or v->u) based on time.
    Calculates:
      - Euclidean distance of the gap (Feature 7)
      - dx and dy of the gap (Feature 8)
    """
    # Pre-process start/end points and times
    stroke_info = []
    has_time = (len(strokes) > 0 and len(strokes[0]) > 0 and len(strokes[0][0]) >= 3)

    for i, s in enumerate(strokes):
        pts = np.array(s)
        if len(pts) == 0:
            stroke_info.append(None)
            continue
            
        # We need the end of the first stroke and start of the second
        start_pt = pts[0]
        end_pt = pts[-1]
        
        if has_time:
            t_start = start_pt[2]
            t_end = end_pt[2]
        else:
            # Fallback to index order
            t_start = float(i)
            t_end = float(i) + 1.0
            
        stroke_info.append({
            "p_start": start_pt[:2],
            "p_end": end_pt[:2], 
            "t_start": t_start,
            "t_end": t_end
        })
        
    feat_7 = []   # Euclidean distance
    feat_8_x = [] # Projected X distance
    feat_8_y = [] # Projected Y distance
    
    for u, v in edges:
        s1 = stroke_info[u]
        s2 = stroke_info[v]
        
        if s1 is None or s2 is None:
            feat_7.append(0.0)
            feat_8_x.append(0.0)
            feat_8_y.append(0.0)
            continue
        
        # Determine order: u -> v OR v -> u
        # We assume the gap is from End(Predecessor) to Start(Successor)
        
        if s1['t_end'] <= s2['t_start']:
            # Sequence: u then v
            p_from = s1['p_end']
            p_to = s2['p_start']
        elif s2['t_end'] <= s1['t_start']:
            # Sequence: v then u
            p_from = s2['p_end']
            p_to = s1['p_start']
        else:
            # Overlapping or ambiguous time: 
            # We typically take the closest endpoints, or default to Index order.
            # Here, we default to u->v if indices are ordered that way, or just u->v raw.
            # To be robust, let's assume u -> v based on index if time fails
            if u < v:
                p_from = s1['p_end']
                p_to = s2['p_start']
            else:
                p_from = s2['p_end']
                p_to = s1['p_start']

        dx = abs(p_to[0] - p_from[0])
        dy = abs(p_to[1] - p_from[1])
        dist = np.sqrt(dx**2 + dy**2)
        
        feat_7.append(float(dist))
        feat_8_x.append(float(dx))
        feat_8_y.append(float(dy))
        
    return {
        "off_stroke_dist": feat_7,
        "off_stroke_dx": feat_8_x,
        "off_stroke_dy": feat_8_y
    }

# 2, 3, 4
def endpoint_and_bbox_distance_features(strokes: list, edges: list[tuple[int, int]]) -> dict:
    """
    Calculates distance metrics related to endpoints and bounding boxes.
    
    Feature 2: Minimum distance between endpoints (start/end) of stroke A and stroke B.
    Feature 3: Maximum distance between endpoints of stroke A and stroke B.
    Feature 4: Euclidean distance between the centers of the Bounding Boxes of stroke A and B.
    """
    # 1. Pre-compute stroke properties (Endpoints and BBox Centers)
    stroke_data = []
    
    for s in strokes:
        pts = np.array(s)[:, :2]
        if len(pts) == 0:
            stroke_data.append(None)
            continue
            
        # Endpoints
        p_start = pts[0]
        p_end = pts[-1]
        
        # Bounding Box Center
        # Center = (Min + Max) / 2
        min_xy = np.min(pts, axis=0)
        max_xy = np.max(pts, axis=0)
        bbox_center = (min_xy + max_xy) / 2.0
        
        stroke_data.append({
            "start": p_start,
            "end": p_end,
            "center": bbox_center
        })
        
    feat_2 = [] # Min endpoint dist
    feat_3 = [] # Max endpoint dist
    feat_4 = [] # BBox center dist
    
    for u, v in edges:
        d1 = stroke_data[u]
        d2 = stroke_data[v]
        
        if d1 is None or d2 is None:
            feat_2.append(0.0)
            feat_3.append(0.0)
            feat_4.append(0.0)
            continue
            
        # --- Features 2 & 3: Endpoint Distances ---
        # There are 4 possible distances between endpoints:
        # Start-Start, Start-End, End-Start, End-End
        
        # We can compute them manually or use cdist on small arrays
        ends_u = np.array([d1['start'], d1['end']])
        ends_v = np.array([d2['start'], d2['end']])
        
        # cdist returns 2x2 matrix
        dists = cdist(ends_u, ends_v)
        
        feat_2.append(float(np.min(dists))) # Feature 2
        feat_3.append(float(np.max(dists))) # Feature 3
        
        # --- Feature 4: BBox Center Distance ---
        c1 = d1['center']
        c2 = d2['center']
        
        center_dist = np.sqrt(np.sum((c1 - c2)**2))
        feat_4.append(float(center_dist))
        
    return {
        "min_dist_endpoints": feat_2, # 2
        "max_dist_endpoints": feat_3, # 3
        "dist_bbox_centers": feat_4   # 4
    }

def extract_stroke_features(strokes: list_strokes, offset, proxy_threshold, time_threshold) -> dict:
    edges = get_edges(strokes, proxy_threshold, time_threshold)

    norm_feats = get_normalization_features(strokes)
    perpen_feats = perpendicularity_features(strokes)
    centroid_dists = centroid_distance_features(strokes, edges)
    bbox_ratios = calculate_bbox_ratios(strokes, edges)
    temp_edge_feats = temporal_edge_features(strokes, edges)

    time_neigh_stats = get_time_neighbor_statistics(strokes, time_threshold)
    space_neigh_stats = get_spatial_neighbor_statistics(strokes, proxy_threshold)

    len_dur_ratios = length_and_duration_ratios(strokes, edges)
    off_stroke_feats = off_stroke_features(strokes, edges)
    end_bbox_feats = endpoint_and_bbox_distance_features(strokes, edges)
    nodes_out = {
            "normalized_width": norm_feats["normalized_width"],   # 12
            "normalized_height": norm_feats["normalized_height"],     # 13
            "linearity_ratio": linearity_ratio(strokes),       # 8
            "accumulated curvature": accumulated_curvature(strokes), # 9

            "num_temporal_neighbours": number_of_temporal_neighbors(strokes, time_threshold), # 14
            "num_spatial_neighbours": number_of_spatial_neighbors(strokes, proxy_threshold), # 15
            "trajectory_length": trajectory_length(strokes), # 1
            "trajectory_duration": trajectory_duration(strokes), # 3
            "area_convex_hull": area_of_convex_hull(strokes), # 2

            "principal_axis_ratio": principal_axis_ratio(strokes), # 4
            "rectangularity": rectangularity(strokes), # 5

            "accumulated_squared_perpendicularity": perpen_feats["squared_perpendicularity"], # 10
            "accumulated_signed_perpendicularity": perpen_feats["signed_perpendicularity"], # 11
            "circular_variance": circular_variance(strokes), # 6
            "normalized_offset_along_principal_axis": principal_axis_features(strokes), # 7

            "avg_dist_time_neighbors": time_neigh_stats["avg_dist_time_neighbors"], # 16
            "std_dist_time_neighbors": time_neigh_stats["std_dist_time_neighbors"], # 17
            "avg_len_time_neighbors": time_neigh_stats["avg_len_time_neighbors"],   # 18
            "std_len_time_neighbors": time_neigh_stats["std_len_time_neighbors"],   # 19

            "avg_dist_space_neighbors": space_neigh_stats["avg_dist_space_neighbors"], # 20
            "std_dist_space_neighbors": space_neigh_stats["std_dist_space_neighbors"], # 21
            "avg_len_space_neighbors": space_neigh_stats["avg_len_space_neighbors"],   # 22
            "std_len_space_neighbors": space_neigh_stats["std_len_space_neighbors"],   # 23
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
            "ratio_curvatures": ratio_of_curvatures(strokes, edges),           # 19

            "ratio_length": len_dur_ratios["ratio_length"],       # 17
            "ratio_duration": len_dur_ratios["ratio_duration"],   # 18

            "off_stroke_dist": off_stroke_feats["off_stroke_dist"], # 7
            "off_stroke_dx": off_stroke_feats["off_stroke_dx"],     # 8 (X)
            "off_stroke_dy": off_stroke_feats["off_stroke_dy"],     # 8 (Y)

            "min_dist_endpoints": end_bbox_feats["min_dist_endpoints"], # 2
            "max_dist_endpoints": end_bbox_feats["max_dist_endpoints"], # 3
            "dist_bbox_centers": end_bbox_feats["dist_bbox_centers"],   # 4
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
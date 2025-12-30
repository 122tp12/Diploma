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
        if not stroke:
            length.append(0.0)
            continue

        # reconstruct absolute points similarly to _to_absolute_points to compute accurate trajectory length
        if len(stroke) == 1:
            length.append(0.0)
            continue

        # initial position
        current_x = stroke[0][0]
        current_y = stroke[0][1]
        if len(stroke[0]) > 2:
            current_t = stroke[0][2]
        else:
            current_t = 0.0

        traj_len = 0.0

        if len(stroke) >= 2:
            vx, vy = stroke[1][0], stroke[1][1]
            current_x += vx
            current_y += vy
            prev_x, prev_y = current_x, current_y

            i = 2
            while i < len(stroke):
                inc = stroke[i]
                ix = inc[0]
                iy = inc[1]
                vx += ix
                vy += iy
                current_x += vx
                current_y += vy
                traj_len += float(np.hypot(current_x - prev_x, current_y - prev_y))
                prev_x, prev_y = current_x, current_y
                i += 1

        length.append(traj_len)

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

def normalized_height(strokes: list_strokes) -> list[float]:
    heights = []

    for stroke in strokes:
        if len(stroke) == 0:
            heights.append(0.0)
            continue
        height = max(point[1] for point in stroke) - min(point[1] for point in stroke)
        heights.append(height)

    median_h = float(np.median(heights)) if len(heights) > 0 else 0.0
    if median_h == 0:
        return [0.0 for _ in heights]

    return [h / median_h for h in heights]

def normalized_width(strokes: list_strokes) -> list[float]:
    widths = []

    for stroke in strokes:
        if len(stroke) == 0:
            widths.append(0.0)
            continue
        width = max(point[0] for point in stroke) - min(point[0] for point in stroke)
        widths.append(width)

    median_h = float(np.median([max(point[1] for point in s) - min(point[1] for point in s) if len(s)>0 else 0.0 for s in strokes])) if len(strokes)>0 else 0.0
    if median_h == 0:
        return [0.0 for _ in widths]

    return [w / median_h for w in widths]

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

# Helpers to convert stroke point representation (as used in parse_inkml/plot)
def _to_absolute_points(stroke: list[list[float]]) -> list[tuple[float, float, float, float]]:
    if not stroke:
        return []

    # First point contains initial x,y,t,f
    current_x = stroke[0][0]
    current_y = stroke[0][1]
    current_t = stroke[0][2] if len(stroke[0]) > 2 else 0.0
    current_f = stroke[0][3] if len(stroke[0]) > 3 else 0.0

    pts = [(current_x, current_y, current_t, current_f)]
    if len(stroke) < 2:
        return pts

    # points[1] is initial velocity; subsequent points are increments to velocity
    vx, vy, vt, vf = stroke[1][0], stroke[1][1], (stroke[1][2] if len(stroke[1])>2 else 0.0), (stroke[1][3] if len(stroke[1])>3 else 0.0)

    current_x += vx
    current_y += vy
    current_t += vt
    current_f += vf
    pts.append((current_x, current_y, current_t, current_f))

    i = 2
    while i < len(stroke):
        inc = stroke[i]
        ix = inc[0]
        iy = inc[1]
        it = inc[2] if len(inc) > 2 else 0.0
        if len(inc) > 3:
            inf = inc[3]
        else:
            inf = 0.0

        vx += ix
        vy += iy
        vt += it
        vf += inf

        current_x += vx
        current_y += vy
        current_t += vt
        current_f += vf
        pts.append((current_x, current_y, current_t, current_f))
        i += 1

    return pts

def _bbox_of_points(pts: list[tuple[float,float,float,float]]):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return xmin, ymin, xmax, ymax

def _centroid_of_points(pts: list[tuple[float,float,float,float]]):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (sum(xs) / len(xs), sum(ys) / len(ys)) if pts else (0.0, 0.0)

def _stroke_total_length(pts: list[tuple[float,float,float,float]]) -> float:
    if len(pts) < 2:
        return 0.0
    s = 0.0
    for i in range(len(pts)-1):
        x1,y1 = pts[i][0], pts[i][1]
        x2,y2 = pts[i+1][0], pts[i+1][1]
        s += float(np.sqrt((x2-x1)**2 + (y2-y1)**2))
    return s

def _stroke_duration(pts: list[tuple[float,float,float,float]]) -> float:
    if not pts:
        return 0.0
    return abs(pts[-1][2] - pts[0][2])

def _stroke_curvature(pts: list[tuple[float,float,float,float]]) -> float:
    # Sum of absolute turning angles between consecutive segments
    if len(pts) < 3:
        return 0.0
    total = 0.0
    for i in range(1, len(pts)-1):
        x0,y0 = pts[i-1][0], pts[i-1][1]
        x1,y1 = pts[i][0], pts[i][1]
        x2,y2 = pts[i+1][0], pts[i+1][1]

        v1 = (x1-x0, y1-y0)
        v2 = (x2-x1, y2-y1)
        n1 = np.hypot(v1[0], v1[1])
        n2 = np.hypot(v2[0], v2[1])
        if n1 == 0 or n2 == 0:
            continue
        dot = (v1[0]*v2[0] + v1[1]*v2[1]) / (n1*n2)
        dot = max(-1.0, min(1.0, dot))
        angle = float(np.arccos(dot))
        total += abs(angle)
    return total

def _convex_hull_area(pts: list[tuple[float,float,float,float]]) -> float:
    # Monotone chain convex hull, then polygon area (shoelace)
    if len(pts) < 3:
        return 0.0
    points = sorted([(p[0], p[1]) for p in pts])

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    # shoelace
    area = 0.0
    for i in range(len(hull)):
        x1,y1 = hull[i]
        x2,y2 = hull[(i+1)%len(hull)]
        area += x1*y2 - x2*y1
    return abs(area) * 0.5

def _pca_eigvals(pts: list[tuple[float,float,float,float]]):
    if len(pts) < 2:
        return (0.0, 0.0)
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    X = np.vstack([xs - xs.mean(), ys - ys.mean()])
    cov = np.cov(X)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        eigvals, eigvecs = np.linalg.eig(cov)
    except Exception:
        return (0.0, 0.0)
    eigvals = np.real(eigvals)
    eigvals_sorted = np.sort(eigvals)[::-1]
    if eigvals_sorted.size == 0:
        return (0.0, 0.0)
    if eigvals_sorted.size == 1:
        return (float(eigvals_sorted[0]), 0.0)
    return (float(eigvals_sorted[0]), float(eigvals_sorted[1]))

def min_endpoint_distance(strokes: list_strokes, edges:list[tuple[int,int]]) -> list[float]:
    distances = []
    for a,b in edges:
        pts_a = _to_absolute_points(strokes[a])
        pts_b = _to_absolute_points(strokes[b])
        if not pts_a or not pts_b:
            distances.append(0.0)
            continue
        ends_a = [pts_a[0], pts_a[-1]]
        ends_b = [pts_b[0], pts_b[-1]]
        min_d = float('inf')
        for pa in ends_a:
            for pb in ends_b:
                d = float(np.hypot(pb[0]-pa[0], pb[1]-pa[1]))
                if d < min_d:
                    min_d = d
        distances.append(min_d)
    return distances

def max_endpoint_distance(strokes: list_strokes, edges:list[tuple[int,int]]) -> list[float]:
    distances = []
    for a,b in edges:
        pts_a = _to_absolute_points(strokes[a])
        pts_b = _to_absolute_points(strokes[b])
        if not pts_a or not pts_b:
            distances.append(0.0)
            continue
        ends_a = [pts_a[0], pts_a[-1]]
        ends_b = [pts_b[0], pts_b[-1]]
        max_d = 0.0
        for pa in ends_a:
            for pb in ends_b:
                d = float(np.hypot(pb[0]-pa[0], pb[1]-pa[1]))
                if d > max_d:
                    max_d = d
        distances.append(max_d)
    return distances

def bbox_centers_distance(strokes: list_strokes, edges:list[tuple[int,int]]) -> list[float]:
    distances = []
    for a,b in edges:
        pts_a = _to_absolute_points(strokes[a])
        pts_b = _to_absolute_points(strokes[b])
        if not pts_a or not pts_b:
            distances.append(0.0)
            continue
        xa, ya = _centroid_of_points(pts_a)
        xb, yb = _centroid_of_points(pts_b)
        distances.append(float(np.hypot(xa-xb, ya-yb)))
    return distances

def horiz_vert_centroid_distances(strokes: list_strokes, edges:list[tuple[int,int]]) -> tuple[list[float], list[float]]:
    horiz = []
    vert = []
    for a,b in edges:
        pts_a = _to_absolute_points(strokes[a])
        pts_b = _to_absolute_points(strokes[b])
        if not pts_a or not pts_b:
            horiz.append(0.0); vert.append(0.0)
            continue
        xa, ya = _centroid_of_points(pts_a)
        xb, yb = _centroid_of_points(pts_b)
        horiz.append(abs(xa-xb))
        vert.append(abs(ya-yb))
    return horiz, vert

def offstroke_axis_distances(strokes: list_strokes, edges:list[tuple[int,int]]) -> tuple[list[float], list[float]]:
    # minimal absolute difference on X and Y between points of two strokes
    dxs = []
    dys = []
    for a,b in edges:
        pts_a = _to_absolute_points(strokes[a])
        pts_b = _to_absolute_points(strokes[b])
        if not pts_a or not pts_b:
            dxs.append(0.0); dys.append(0.0)
            continue
        min_dx = float('inf')
        min_dy = float('inf')
        for pa in pts_a:
            for pb in pts_b:
                min_dx = min(min_dx, abs(pa[0]-pb[0]))
                min_dy = min(min_dy, abs(pa[1]-pb[1]))
        dxs.append(min_dx)
        dys.append(min_dy)
    return dxs, dys

def temporal_distance_between_strokes(strokes: list_strokes, edges:list[tuple[int,int]]) -> list[float]:
    distances = []
    for a,b in edges:
        pts_a = _to_absolute_points(strokes[a])
        pts_b = _to_absolute_points(strokes[b])
        if not pts_a or not pts_b:
            distances.append(0.0)
            continue
        # use start times difference
        ta = pts_a[0][2]
        tb = pts_b[0][2]
        distances.append(abs(ta-tb))
    return distances

def ratio_offstroke_to_temporal(min_offstroke: list[float], temporal: list[float]) -> list[float]:
    ratios = []
    for off, t in zip(min_offstroke, temporal):
        if t == 0.0:
            ratios.append(0.0)
        else:
            ratios.append(off / t)
    return ratios

def bbox_area_ratios_and_dimensions(strokes: list_strokes, edges:list[tuple[int,int]]):
    area_ratios = []
    width_ratios = []
    height_ratios = []
    diag_ratios = []
    for a,b in edges:
        pts_a = _to_absolute_points(strokes[a])
        pts_b = _to_absolute_points(strokes[b])
        if not pts_a or not pts_b:
            area_ratios.append(0.0); width_ratios.append(0.0); height_ratios.append(0.0); diag_ratios.append(0.0)
            continue
        xa_min, ya_min, xa_max, ya_max = _bbox_of_points(pts_a)
        xb_min, yb_min, xb_max, yb_max = _bbox_of_points(pts_b)
        wa = xa_max - xa_min
        ha = ya_max - ya_min
        wb = xb_max - xb_min
        hb = yb_max - yb_min
        area_a = wa * ha
        area_b = wb * hb
        union_xmin = min(xa_min, xb_min)
        union_xmax = max(xa_max, xb_max)
        union_ymin = min(ya_min, yb_min)
        union_ymax = max(ya_max, yb_max)
        union_area = max(1e-9, (union_xmax-union_xmin) * (union_ymax-union_ymin))
        area_ratios.append(max(area_a, area_b) / union_area)
        width_ratios.append(wa / wb if wb != 0 else 0.0)
        height_ratios.append(ha / hb if hb != 0 else 0.0)
        da = float(np.hypot(wa, ha))
        db = float(np.hypot(wb, hb))
        diag_ratios.append(da / db if db != 0 else 0.0)
    return area_ratios, width_ratios, height_ratios, diag_ratios

def ratio_lengths_durations_curvatures(strokes: list_strokes, edges:list[tuple[int,int]]):
    ratio_lengths = []
    ratio_durations = []
    ratio_curvatures = []
    for a,b in edges:
        pts_a = _to_absolute_points(strokes[a])
        pts_b = _to_absolute_points(strokes[b])
        if not pts_a or not pts_b:
            ratio_lengths.append(0.0); ratio_durations.append(0.0); ratio_curvatures.append(0.0)
            continue
        la = _stroke_total_length(pts_a)
        lb = _stroke_total_length(pts_b)
        ratio_lengths.append(la / lb if lb != 0 else 0.0)

        da = _stroke_duration(pts_a)
        db = _stroke_duration(pts_b)
        ratio_durations.append(da / db if db != 0 else 0.0)

        ca = _stroke_curvature(pts_a)
        cb = _stroke_curvature(pts_b)
        ratio_curvatures.append(ca / cb if cb != 0 else 0.0)
    return ratio_lengths, ratio_durations, ratio_curvatures

def straightness_ratio(strokes: list_strokes) -> list[float]:
    ratios = []

    for stroke in strokes:
        if len(stroke) < 2:
            ratios.append(0.0)
            continue

        x1, y1 = stroke[0][0], stroke[0][1]
        x2, y2 = stroke[-1][0], stroke[-1][1]
        first_last_dist = float(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))

        traj_len = 0.0
        for i in range(len(stroke) - 1):
            xa, ya = stroke[i][0], stroke[i][1]
            xb, yb = stroke[i+1][0], stroke[i+1][1]
            traj_len += float(np.sqrt((xb - xa)**2 + (yb - ya)**2))

        if traj_len == 0.0:
            ratios.append(0.0)
        else:
            ratios.append(first_last_dist / traj_len)

    return ratios

def num_spatial_neighbors(strokes: list_strokes, edges: list[tuple[int, int]]) -> list[int]:
    num_nodes = len(strokes)
    neighbors = [set() for _ in range(num_nodes)]

    for a, b in edges:
        if 0 <= a < num_nodes and 0 <= b < num_nodes and a != b:
            neighbors[a].add(b)

    return [len(s) for s in neighbors]

def node_geometric_features(strokes: list_strokes, edges: list[tuple[int,int]]) -> dict:
    n = len(strokes)
    traj_lengths = stroke_length(strokes)
    ch_areas = []
    durations = []
    pca_ratios = []
    rects = []
    circ_vars = []
    centroid_offset = []
    acc_curv = []
    acc_sq_perp = []
    acc_signed_perp = []

    # precompute median stroke height for normalization
    heights = [max(p[1] for p in s) - min(p[1] for p in s) if len(s)>0 else 0.0 for s in strokes]
    median_h = float(np.median(heights)) if n>0 else 0.0

    for idx, stroke in enumerate(strokes):
        pts = _to_absolute_points(stroke)
        if not pts:
            ch_areas.append(0.0)
            durations.append(0.0)
            pca_ratios.append(0.0)
            rects.append(0.0)
            circ_vars.append(0.0)
            centroid_offset.append(0.0)
            acc_curv.append(0.0)
            acc_sq_perp.append(0.0)
            acc_signed_perp.append(0.0)
            continue

        # convex hull area
        ch_area = _convex_hull_area(pts)
        ch_areas.append(ch_area)

        # duration
        durations.append(_stroke_duration(pts))

        # PCA eigenvalues ratio (major/minor)
        e1,e2 = _pca_eigvals(pts)
        if e2 == 0:
            pca_ratios.append(0.0)
        else:
            pca_ratios.append(e1 / e2)

        # oriented bounding box via PCA: rotate points to principal axis and compute bbox area
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        
        # compute PCA eigenvectors for later use
        cx, cy = xs.mean(), ys.mean()
        cov = np.cov(np.vstack([xs - cx, ys - cy]))
        cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            evals, evecs = np.linalg.eig(cov)
            evals = np.real(evals)
            evecs = np.real(evecs)
        except Exception:
            evals = np.array([0.0, 0.0])
            evecs = np.array([[1.0, 0.0], [0.0, 1.0]])
        
        if e1 == 0 and e2 == 0:
            rects.append(0.0)
        else:
            # principal direction = eigenvector of largest eigenvalue
            v = evecs[:, np.argmax(evals)]
            # rotation matrix to align v with x-axis
            angle = np.arctan2(v[1], v[0])
            c = np.cos(-angle); s = np.sin(-angle)
            xr = c*(xs-cx) - s*(ys-cy)
            yr = s*(xs-cx) + c*(ys-cy)
            w = xr.max() - xr.min()
            h = yr.max() - yr.min()
            obb_area = w * h
            rects.append((ch_area / obb_area) if obb_area != 0 else 0.0)

        # circular variance
        cenx, ceny = _centroid_of_points(pts)
        angles = [np.arctan2(p[1]-ceny, p[0]-cenx) for p in pts]
        R = np.hypot(np.sum(np.cos(angles)), np.sum(np.sin(angles))) / len(angles)
        circ_vars.append(1.0 - R)

        # normalized centroid offset along principal axis
        # project centroid relative to stroke mean on principal axis and normalize by sqrt(e1)
        if e1 + e2 == 0:
            centroid_offset.append(0.0)
        else:
            centroid = (cenx, ceny)
            meanx, meany = xs.mean(), ys.mean()
            v = evecs[:, np.argmax(evals)]
            proj = (centroid[0]-meanx)*v[0] + (centroid[1]-meany)*v[1]
            norm = np.sqrt(e1) if e1>0 else 1.0
            centroid_offset.append(proj / norm)

        # accumulated curvature
        acc_curv.append(_stroke_curvature(pts))

        # accumulated perpendicularity measures
        sq_perp = 0.0
        signed_perp = 0.0
        for i in range(1, len(pts)-1):
            x0,y0 = pts[i-1][0], pts[i-1][1]
            x1,y1 = pts[i][0], pts[i][1]
            x2,y2 = pts[i+1][0], pts[i+1][1]
            v1 = (x1-x0, y1-y0)
            v2 = (x2-x1, y2-y1)
            n1 = np.hypot(v1[0], v1[1])
            n2 = np.hypot(v2[0], v2[1])
            if n1 == 0 or n2 == 0:
                continue
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            val = cross / (n1*n2)
            sq_perp += val*val
            signed_perp += val

        acc_sq_perp.append(sq_perp)
        acc_signed_perp.append(signed_perp)

    # neighbor-based stats: temporal neighbors
    # compute start times for strokes
    start_times = []
    centroids = []
    lengths = traj_lengths
    for stroke in strokes:
        pts = _to_absolute_points(stroke)
        if not pts:
            start_times.append(0.0)
            centroids.append((0.0,0.0))
        else:
            start_times.append(pts[0][2])
            centroids.append(_centroid_of_points(pts))

    sorted_times = sorted(start_times)
    if len(sorted_times) >= 2:
        gaps = [sorted_times[i+1]-sorted_times[i] for i in range(len(sorted_times)-1)]
        median_gap = float(np.median(gaps)) if gaps else 0.0
    else:
        median_gap = 0.0
    time_threshold = median_gap * 1.5 if median_gap > 0 else 1.0

    num_temp_neighbors = []
    avg_dist_time_neighbors = []
    std_dist_time_neighbors = []
    avg_len_time_neighbors = []
    std_len_time_neighbors = []

    # spatial neighbors sets
    spatial_neighbors = [set() for _ in range(n)]
    for a,b in edges:
        if 0 <= a < n and 0 <= b < n and a != b:
            spatial_neighbors[a].add(b)

    num_space_neighbors = [len(s) for s in spatial_neighbors]
    avg_dist_space_neighbors = []
    std_dist_space_neighbors = []
    avg_len_space_neighbors = []
    std_len_space_neighbors = []

    for i in range(n):
        # temporal neighbors
        t_neigh = [j for j in range(n) if j!=i and abs(start_times[j]-start_times[i]) <= time_threshold]
        num_temp_neighbors.append(len(t_neigh))
        if t_neigh:
            dists = [float(np.hypot(centroids[i][0]-centroids[j][0], centroids[i][1]-centroids[j][1])) for j in t_neigh]
            avg_dist_time_neighbors.append(float(np.mean(dists)))
            std_dist_time_neighbors.append(float(np.std(dists)))
            lens = [lengths[j] for j in t_neigh]
            avg_len_time_neighbors.append(float(np.mean(lens)))
            std_len_time_neighbors.append(float(np.std(lens)))
        else:
            avg_dist_time_neighbors.append(0.0); std_dist_time_neighbors.append(0.0)
            avg_len_time_neighbors.append(0.0); std_len_time_neighbors.append(0.0)

        # spatial neighbors
        s_neigh = list(spatial_neighbors[i])
        if s_neigh:
            dists = [float(np.hypot(centroids[i][0]-centroids[j][0], centroids[i][1]-centroids[j][1])) for j in s_neigh]
            avg_dist_space_neighbors.append(float(np.mean(dists)))
            std_dist_space_neighbors.append(float(np.std(dists)))
            lens = [lengths[j] for j in s_neigh]
            avg_len_space_neighbors.append(float(np.mean(lens)))
            std_len_space_neighbors.append(float(np.std(lens)))
        else:
            avg_dist_space_neighbors.append(0.0); std_dist_space_neighbors.append(0.0)
            avg_len_space_neighbors.append(0.0); std_len_space_neighbors.append(0.0)

    return {
        "trajectory_length": traj_lengths,
        "convex_hull_area": ch_areas,
        "duration": durations,
        "pca_ratio": pca_ratios,
        "rectangularity": rects,
        "circular_variance": circ_vars,
        "centroid_offset": centroid_offset,
        "accumulated_curvature": acc_curv,
        "accum_squared_perp": acc_sq_perp,
        "accum_signed_perp": acc_signed_perp,
        "width_norm": normalized_width(strokes),
        "height_norm": normalized_height(strokes),
        "num_temporal_neighbors": num_temp_neighbors,
        "num_spatial_neighbors": num_space_neighbors,
        "avg_dist_time_neighbors": avg_dist_time_neighbors,
        "std_dist_time_neighbors": std_dist_time_neighbors,
        "avg_len_time_neighbors": avg_len_time_neighbors,
        "std_len_time_neighbors": std_len_time_neighbors,
        "avg_dist_space_neighbors": avg_dist_space_neighbors,
        "std_dist_space_neighbors": std_dist_space_neighbors,
        "avg_len_space_neighbors": avg_len_space_neighbors,
        "std_len_space_neighbors": std_len_space_neighbors
    }

# Main function to extract all features
def extract_stroke_features(strokes: list_strokes) -> dict:
    edges=get_edges(strokes, threshold=10.0)
    
    node_feats = node_geometric_features(strokes, edges)
    return {
        "nodes":{
            "trajectory_length": node_feats["trajectory_length"],
            "convex_hull_area": node_feats["convex_hull_area"],
            "duration": node_feats["duration"],
            "pca_ratio": node_feats["pca_ratio"],
            "rectangularity": node_feats["rectangularity"],
            "circular_variance": node_feats["circular_variance"],
            "centroid_offset": node_feats["centroid_offset"],
            "accumulated_curvature": node_feats["accumulated_curvature"],
            "accum_squared_perp": node_feats["accum_squared_perp"],
            "accum_signed_perp": node_feats["accum_signed_perp"],
            "width_norm": node_feats["width_norm"],
            "height_norm": node_feats["height_norm"],
            "num_temporal_neighbors": node_feats["num_temporal_neighbors"],
            "num_spatial_neighbors": node_feats["num_spatial_neighbors"],
            "avg_dist_time_neighbors": node_feats["avg_dist_time_neighbors"],
            "std_dist_time_neighbors": node_feats["std_dist_time_neighbors"],
            "avg_len_time_neighbors": node_feats["avg_len_time_neighbors"],
            "std_len_time_neighbors": node_feats["std_len_time_neighbors"],
            "avg_dist_space_neighbors": node_feats["avg_dist_space_neighbors"],
            "std_dist_space_neighbors": node_feats["std_dist_space_neighbors"],
            "avg_len_space_neighbors": node_feats["avg_len_space_neighbors"],
            "std_len_space_neighbors": node_feats["std_len_space_neighbors"]
        },
        "edge_index": edges,
        "edges_features": {
            "min_distance": min_distance_between_strokes(strokes, edges),
            "min_endpoint_distance": min_endpoint_distance(strokes, edges),
            "max_endpoint_distance": max_endpoint_distance(strokes, edges),
            "bbox_centers_distance": bbox_centers_distance(strokes, edges),
            "centroid_dx": horiz_vert_centroid_distances(strokes, edges)[0],
            "centroid_dy": horiz_vert_centroid_distances(strokes, edges)[1],
            "offstroke_dx": offstroke_axis_distances(strokes, edges)[0],
            "offstroke_dy": offstroke_axis_distances(strokes, edges)[1],
            "temporal_distance": temporal_distance_between_strokes(strokes, edges),
            "ratio_offstroke_to_temporal": ratio_offstroke_to_temporal(min_distance_between_strokes(strokes, edges), temporal_distance_between_strokes(strokes, edges)),
            "ratio_offstrokex_to_temporal": ratio_offstroke_to_temporal(offstroke_axis_distances(strokes, edges)[0], temporal_distance_between_strokes(strokes, edges)),
            "ratio_offstrokey_to_temporal": ratio_offstroke_to_temporal(offstroke_axis_distances(strokes, edges)[1], temporal_distance_between_strokes(strokes, edges)),
            "bbox_area_ratio": bbox_area_ratios_and_dimensions(strokes, edges)[0],
            "bbox_width_ratio": bbox_area_ratios_and_dimensions(strokes, edges)[1],
            "bbox_height_ratio": bbox_area_ratios_and_dimensions(strokes, edges)[2],
            "bbox_diag_ratio": bbox_area_ratios_and_dimensions(strokes, edges)[3],
            "ratio_lengths": ratio_lengths_durations_curvatures(strokes, edges)[0],
            "ratio_durations": ratio_lengths_durations_curvatures(strokes, edges)[1],
            "ratio_curvatures": ratio_lengths_durations_curvatures(strokes, edges)[2]
        }
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
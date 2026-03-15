import numpy as np


class RandomMarkedStatesGenerator:

    def generate_multiple_paths(
        self,
        num_paths: int,
        voronoi_skeleton: np.ndarray,
        distance_field: np.ndarray,
        min_clearance: float = 5.0,
        min_distance: int = 20,
        endpoint_exclusion_radius: float = None
    ) -> list[list[tuple[float, float]]]:
        all_paths = []
        used_endpoints = []
        
        if endpoint_exclusion_radius is None:
            height, width = voronoi_skeleton.shape
            endpoint_exclusion_radius = min(height, width) * 0.05
        
        max_attempts = num_paths * 3
        attempts = 0
        
        while len(all_paths) < num_paths and attempts < max_attempts:
            attempts += 1
            
            path = self.generate_random_marked_states(
                voronoi_skeleton,
                distance_field,
                min_clearance=min_clearance,
                min_distance=min_distance,
                exclude_endpoints=used_endpoints,
                endpoint_exclusion_radius=endpoint_exclusion_radius
            )
            
            if path and len(path) >= 2:
                is_duplicate = False
                for existing_path in all_paths:
                    start_dist = np.sqrt((path[0][0] - existing_path[0][0])**2 + (path[0][1] - existing_path[0][1])**2)
                    end_dist = np.sqrt((path[-1][0] - existing_path[-1][0])**2 + (path[-1][1] - existing_path[-1][1])**2)
                    if start_dist < endpoint_exclusion_radius or end_dist < endpoint_exclusion_radius:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    all_paths.append(path)
                    used_endpoints.append(path[0])
                    used_endpoints.append(path[-1])
        
        return all_paths

    def generate_random_marked_states(
        self, voronoi_skeleton: np.ndarray, 
        distance_field: np.ndarray,
        min_clearance: float = 5.0,
        min_distance: int = 20,
        exclude_endpoints: list[tuple[float, float]] = None,
        endpoint_exclusion_radius: float = None
    ) -> list[tuple[float, float]]:
        if voronoi_skeleton is None or distance_field is None:
            return []

        height, width = voronoi_skeleton.shape
        skeleton_points = np.argwhere(voronoi_skeleton == 1)

        if len(skeleton_points) == 0:
            return []

        if exclude_endpoints is None:
            exclude_endpoints = []

        if endpoint_exclusion_radius is None:
            endpoint_exclusion_radius = min(height, width) * 0.05

        exclude_endpoints_rc = [(int(height - y), int(x)) for x, y in exclude_endpoints]

        def is_endpoint_excluded(row, col):
            for er, ec in exclude_endpoints_rc:
                if np.sqrt((row - er)**2 + (col - ec)**2) < endpoint_exclusion_radius:
                    return True
            return False

        def is_valid_point(row, col):
            if row < 0 or row >= height or col < 0 or col >= width:
                return False
            if distance_field[row, col] < min_clearance:
                return False
            return True

        def get_neighbors(row, col):
            neighbors = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        if voronoi_skeleton[nr, nc] == 1:
                            neighbors.append((nr, nc))
            return neighbors

        valid_skeleton = [(r, c) for r, c in skeleton_points if is_valid_point(r, c)]

        if len(valid_skeleton) == 0:
            return []

        edge_margin = int(min(height, width) * 0.15)

        top_edge = [(r, c) for r, c in valid_skeleton if r < edge_margin and not is_endpoint_excluded(r, c)]
        bottom_edge = [(r, c) for r, c in valid_skeleton if r > height - edge_margin and not is_endpoint_excluded(r, c)]
        left_edge = [(r, c) for r, c in valid_skeleton if c < edge_margin and not is_endpoint_excluded(r, c)]
        right_edge = [(r, c) for r, c in valid_skeleton if c > width - edge_margin and not is_endpoint_excluded(r, c)]

        edges = {
            'top': top_edge,
            'bottom': bottom_edge,
            'left': left_edge,
            'right': right_edge
        }

        opposite = {
            'top': 'bottom',
            'bottom': 'top',
            'left': 'right',
            'right': 'left'
        }

        valid_edges = [(name, pts) for name, pts in edges.items() if len(pts) > 0]

        if len(valid_edges) < 2:
            return []

        start_edge_name, start_candidates = valid_edges[np.random.randint(len(valid_edges))]
        start_point = start_candidates[np.random.randint(len(start_candidates))]

        end_edge_name = opposite[start_edge_name]
        end_candidates = edges[end_edge_name]

        if len(end_candidates) == 0:
            other_edges = [(n, p) for n, p in valid_edges if n != start_edge_name and len(p) > 0]
            if not other_edges:
                return []
            end_edge_name, end_candidates = other_edges[np.random.randint(len(other_edges))]

        end_point = end_candidates[np.random.randint(len(end_candidates))]

        from collections import deque

        def bfs_path(start, end):
            queue = deque([(start, [start])])
            visited = {start}
            
            while queue:
                current, path = queue.popleft()
                
                if current == end:
                    return path
                
                for neighbor in get_neighbors(current[0], current[1]):
                    if neighbor not in visited and is_valid_point(neighbor[0], neighbor[1]):
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
            
            return None

        full_path = bfs_path(start_point, end_point)

        if full_path is None or len(full_path) < 2:
            return []

        selected = [full_path[0]]

        for i in range(1, len(full_path)):
            pt = full_path[i]
            last_selected = selected[-1]
            dist = np.sqrt((pt[0] - last_selected[0])**2 + (pt[1] - last_selected[1])**2)
            
            if dist >= min_distance:
                selected.append(pt)

        if full_path[-1] not in selected:
            last_selected = selected[-1]
            end_pt = full_path[-1]
            dist_to_end = np.sqrt((end_pt[0] - last_selected[0])**2 + (end_pt[1] - last_selected[1])**2)
            
            if dist_to_end < min_distance and len(selected) > 1:
                selected[-1] = end_pt
            else:
                selected.append(end_pt)

        marked_states = [(float(col), float(height - row)) for row, col in selected]

        return marked_states

from control.AGV import AGV
from control.PathCreationAlgorithm import PathCreationAlgorithm
from control.CollisionSectorAlgorithm import CollisionSectorAlgorithm
from .Resource import Resource
import copy
import numpy as np

from itertools import combinations

class RAM:

    def __init__(self):
        self.global_resources = {}
        self.resource_counter = 0

    def register_collision_pair(self, res_id, agv1_id, agv2_id):
        if res_id not in self.global_resources:
            self.global_resources[res_id] = Resource(res_id, agv1_id, agv2_id)
        return res_id
    
class StageTransitionControl:

    def __init__(self):
        self.agvs = []
        self.col_sectors = []
        self.ram = RAM()

        self.path_creator = PathCreationAlgorithm()
        self.col_det_alg = CollisionSectorAlgorithm()

    def create_paths(self) -> None:
        for agv in self.agvs:
            if agv.path == []:
                path = self.path_creator.create_path(agv.marked_states.copy(), agv.orientation, agv.radius)
                agv.path = path

    def detec_col_sectors(self):
        self.ram = RAM()
        for agv in self.agvs:
            agv.path_sectors = {}
            
        for agv1, agv2 in combinations(self.agvs, 2):
            for i in range(len(agv1.path)):
                for j in range(len(agv2.path)):
                    curveA = agv1.path[i]
                    curveB = agv2.path[j]

                    s1_list, s2_list = self.col_det_alg.process_curve_pair_multi(
                        agv1.id, i, curveA, agv2.id, j, curveB,
                        agv2.radius, agv1.radius, emergency_factor=1.1
                    )

                    for s1, s2 in zip(s1_list, s2_list):
                        for rid in s1.resource_ids:
                            self.ram.register_collision_pair(rid, agv1.id, agv2.id)
                            agv1.add_sector_to_curve(i, s1)
                            agv2.add_sector_to_curve(j, s2)

    def process_agv_step(self, agv):
        current_sectors = agv.get_current_curve_sectors()

        # print(f"AGV{agv.id}: t={agv.state.current_t:.2f}, curve={agv.state.current_curve_idx}, "
        #     f"status={agv.state.status}, R={agv.state.R}, PH={agv.state.PH}")
        
        if agv.state.is_inside_owned_sector(current_sectors):
            agv.state.status = "running"
            self._check_release(agv, current_sectors)
            return
        
        is_inside, sector = agv.state.is_inside_any_sector(current_sectors)
        if is_inside and not all(r in agv.state.PH for r in sector.resource_ids):
            agv.state.status = "iddling"
            self._request_resources(agv, sector.resource_ids)
            return

        event, data = agv.state.check_for_events(current_sectors, agv.path_sectors)

        if event == "EVENT_RELEASE":
            self._release_resources(agv, data)
            agv.state.status = "running"

        elif event == "EVENT_GET_ACCESS":
            self._request_resources(agv, data)
            
        elif event == "EVENT_BRAKE":
            agv.state.status = "iddling"

        elif event is None:
            if not agv.state.R or all(r in agv.state.PH for r in agv.state.R):
                agv.state.status = "running"
        
        if agv.state.status == "iddling" and agv.state.R:
            self._try_acquire_resources(agv)

    def _request_resources(self, agv, resource_ids):
        for res_id in resource_ids:
            if res_id not in agv.state.R:
                self.ram.global_resources[res_id].get_access(agv.id)
                agv.state.R.add(res_id)
        
        self._try_acquire_resources(agv)

    def _try_acquire_resources(self, agv):
        pending = [r for r in agv.state.R if r not in agv.state.PH]
        
        if not pending:
            agv.state.status = "running"
            return
        
        can_go_collision = self.check_collision_safety(agv.id, pending)
        can_go_deadlock = self.is_state_safe(agv.id, pending, [])
        
        if can_go_collision and can_go_deadlock:
            agv.state.PH.update(pending)
            agv.state.status = "running"
        else:
            agv.state.status = "iddling"

    def _release_resources(self, agv, resource_ids):
        for res_id in resource_ids:
            if res_id in agv.state.PH:
                self.ram.global_resources[res_id].release(agv.id)
                agv.state.PH.discard(res_id)
                agv.state.R.discard(res_id)

    def _check_release(self, agv, current_sectors):
        event, data = agv.state.check_for_events(current_sectors, agv.path_sectors)
        if event == "EVENT_RELEASE":
            self._release_resources(agv, data)

    def get_agvs_number(self) -> int:
        return len(self.agvs)
    
    def calculate_control_points(self, agv, sector, curve_length):
        braking_dist = (agv.state.max_v ** 2) / (2 * agv.state.max_a)
        delta_t_braking = braking_dist / curve_length
        sector.t_critical = max(0.0, sector.t_l - delta_t_braking)
        sector.t_query = max(0.0, sector.t_critical - 0.1)

    def load_agvs(self, loaded_agvs: dict[str, AGV]) -> None:
        for agv in loaded_agvs.values():
            self.agvs.append(agv)

    def trigger_path_creation(self) -> None:
        self.create_paths()
        for agv in self.agvs:
            agv.init_path_lengths()

    def check_collision_safety(self, robot_id, requested_resource_ids):
        for res_id in requested_resource_ids:
            resource = self.ram.global_resources[res_id]
            if not resource.is_first(robot_id):
                return False
        return True
    
    def can_reach_private_state(self, robot_id, temp_resources_map):
        agv = self.agvs[robot_id]
        current_sectors = agv.get_current_curve_sectors()
        
        if agv.state.in_private_sector(current_sectors):
            return True
        
        future_sectors = agv.state.get_sectors_until_next_private(agv.path_sectors)
        
        for sector in future_sectors:
            for res_id in sector.resource_ids:
                if res_id not in temp_resources_map:
                    continue
                res = temp_resources_map[res_id]
                if not res.is_first(robot_id):
                    return False
        
        return True
    
    def is_state_safe(self, robot_id, res_to_access, res_to_release):
        temp_ram = copy.deepcopy(self.ram)

        for res_id in res_to_access:
            if res_id in temp_ram.global_resources:
                temp_ram.global_resources[res_id].get_access(robot_id)
        for res_id in res_to_release:
            if res_id in temp_ram.global_resources:
                temp_ram.global_resources[res_id].release(robot_id)

        remaining_robots = [agv.id for agv in self.agvs if agv.state.status != "finished"]
        max_iterations = len(remaining_robots) * 2
        iterations = 0
        
        while remaining_robots and iterations < max_iterations:
            iterations += 1
            progress = False
            
            for r_id in remaining_robots[:]:  # Kopia listy do iteracji
                if self.can_reach_private_state(r_id, temp_ram.global_resources):
                    for res in temp_ram.global_resources.values():
                        res.release(r_id)
                    remaining_robots.remove(r_id)
                    progress = True
            
            if not progress:
                break

        return len(remaining_robots) == 0
    
    def calculate_bezier_length(self, verts, steps=100):
        p0, p1, p2 = map(np.array, verts)
        length = 0.0
        prev_pt = p0
        
        for i in range(1, steps + 1):
            t = i / steps
            curr_pt = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2
            length += np.linalg.norm(curr_pt - prev_pt)
            prev_pt = curr_pt
            
        return length
    
    def get_all_control_points(self):
        for agv in self.agvs:
            for curve_idx, sectors in agv.path_sectors.items():
                verts = agv.path[curve_idx]
                curve_len = self.calculate_bezier_length(verts)

                for sector in sectors:
                    self.calculate_control_points(agv, sector, curve_len)

    def finalize_agv_sectors(self):
        for agv in self.agvs:
            for curve_idx in agv.path_sectors:
                raw_sectors = agv.path_sectors[curve_idx]
                merged = self.col_det_alg.merge_sectors(raw_sectors)   
                agv.path_sectors[curve_idx] = merged

    def global_merge(self):
        for agv in self.agvs:
            n = len(agv.path)
            for i in range(n):
                next_i = (i + 1) % n
                if i not in agv.path_sectors or next_i not in agv.path_sectors:
                    continue
                
                curr_s = max(agv.path_sectors[i], key=lambda x: x.t_u) if agv.path_sectors[i] else None
                next_s = min(agv.path_sectors[next_i], key=lambda x: x.t_l) if agv.path_sectors[next_i] else None
                
                if curr_s and next_s:
                    if set(curr_s.resource_ids) & set(next_s.resource_ids) or \
                    (curr_s.t_u > 0.8 and next_s.t_l < 0.2):
                        
                        curr_s.t_u = 1.0
                        next_s.t_l = 0.0
                        
                        union_res = list(set(curr_s.resource_ids + next_s.resource_ids))
                        curr_s.resource_ids = union_res
                        next_s.resource_ids = union_res

    def reset_all(self):
        for agv in self.agvs:
            agv.reset()

        for res_id, res_obj in self.ram.global_resources.items():
            res_obj.priority_list = []
            
    def step_all(self, dt):
        for agv in self.agvs:
            if agv.state.status != "finished":
                self.process_agv_step(agv)
        
        for agv in self.agvs:
            agv.step(dt)

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

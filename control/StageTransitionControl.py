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

class RobotState:

    def __init__(self, robot_id, radius, max_v, max_a):
        self.id = robot_id
        self.radius = radius
        self.max_v = max_v
        self.max_a = max_a
        
        # self.path_segments = path_segments
        self.current_t = 0.0
        self.current_curve_idx = 0
        self.status = "running"

        self.R = set()
        self.PH = set()

    def reset(self):
        self.current_t = 0.0
        self.current_curve_idx = 0
        self.status = "running"
        self.R = set()
        self.PH = set()

    def needs_resources(self):
        return [r for r in self.R if r not in self.PH]
    
    def update_control_points(self, sectors_on_curve, curve_length):
        braking_dist = (self.max_v**2) / (2 * self.max_a)
        delta_t_braking = braking_dist / curve_length
        for sector in sectors_on_curve:
            sector.t_critical = max(0.0, sector.t_l - delta_t_braking)
            sector.t_query = max(0.0, sector.t_critical - 0.05)

    def is_inside_owned_sector(self, sectors_on_curve):
        for sector in sectors_on_curve:
            if sector.t_l <= self.current_t <= sector.t_u:
                if all(res in self.PH for res in sector.resource_ids):
                    return True
        return False

    def check_for_events(self, sectors_on_curve, all_path_sectors):
        num_curves = len(all_path_sectors)
        
        for sector in sectors_on_curve:
            if self.current_t >= sector.t_u - 0.001: 
                is_continued = False
                if sector.t_u >= 0.999:
                    next_idx = (self.current_curve_idx + 1) % num_curves
                    next_sectors = all_path_sectors.get(next_idx, [])
                    for s_next in next_sectors:
                        if s_next.t_l <= 0.001 and set(s_next.resource_ids) == set(sector.resource_ids):
                            is_continued = True
                            break
                
                if not is_continued and any(res in self.PH for res in sector.resource_ids):
                    return "EVENT_RELEASE", sector.resource_ids

            if self.current_t >= sector.t_query and not any(res in self.R for res in sector.resource_ids):
                return "EVENT_GET_ACCESS", sector.resource_ids
            
            if self.current_t >= sector.t_critical and self.current_t < sector.t_u:
                if not all(res in self.PH for res in sector.resource_ids):
                    self.status = "iddling"
                    return "EVENT_BRAKE", None
            
        return None, None
    
    def get_sectors_until_next_private(self, all_path_sectors):
        needed_sectors = []

        for idx in range(self.current_curve_idx, len(all_path_sectors)):
            sectors_on_curve = all_path_sectors[idx]
            sorted_sectors = sorted(sectors_on_curve, key=lambda x: x.t_l)

            for sector in sorted_sectors:
                if idx > self.current_curve_idx or sector.t_l > self.current_t:
                    needed_sectors.append(sector)

            if not any(s.t_l > self.current_t for s in sorted_sectors):
                break

        return needed_sectors
    
    def in_private_sector(self, current_curve_sector):
        for sector in current_curve_sector:
            if sector.t_l <= self.current_t <= sector.t_u:
                return False
        return True
        
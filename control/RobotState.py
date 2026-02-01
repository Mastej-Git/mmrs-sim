class RobotState:

    def __init__(self, robot_id, radius, max_v, max_a):
        self.id = robot_id
        self.radius = radius
        self.max_v = max_v
        self.max_a = max_a
        
        # self.path_segments = path_segments
        self.current_t = 0.0
        self.current_curve_idx = 0
        self.status = "iddling"

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

    def check_for_events(self, sectors_on_curve):
        for sector in sectors_on_curve:
            if self.current_t >= sector.t_query and not any(res in self.R for res in sector.resource_ids):
                self.R.update(sector.resource_ids)
                return "EVENT_GET_ACCESS", sector.resource_ids
            
            if sector.t_query <= self.current_t < sector.t_critical:
                pass

            if self.current_t >= sector.t_critical:
                if not all(res in self.PH for res in sector.resource_ids):
                    self.status = "iddling"
                    return "EVENT_BRAKE", None
                
            if self.current_t > sector.t_u and any(res in self.PH for res in sector.resource_ids):
                released = sector.resource_ids
                self.PH.difference_update(released)
                self.R.difference_update(released)
                return "EVENT_RELEASE", released
            
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
        
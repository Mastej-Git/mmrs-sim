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

    def is_inside_any_sector(self, sectors_on_curve):
        for sector in sectors_on_curve:
            if sector.t_l <= self.current_t <= sector.t_u:
                return True, sector
        return False, None

    def check_for_events(self, sectors_on_curve, all_path_sectors):
        num_curves = len(all_path_sectors)

        events = []
        
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
                
                if not is_continued:
                    resources_to_release = [r for r in sector.resource_ids if r in self.PH]
                    if resources_to_release:
                        events.append(("EVENT_RELEASE", resources_to_release, sector.t_u))

            if self.current_t >= sector.t_query and self.current_t < sector.t_l:
                resources_needed = [r for r in sector.resource_ids if r not in self.PH]
                if resources_needed and not any(r in self.R for r in resources_needed):
                    events.append(("EVENT_GET_ACCESS", sector.resource_ids, sector.t_query))
            
            if self.current_t >= sector.t_critical and self.current_t < sector.t_l:
                if not all(res in self.PH for res in sector.resource_ids):
                    events.append(("EVENT_BRAKE", sector.resource_ids, sector.t_critical))

        for event_type in ["EVENT_RELEASE", "EVENT_GET_ACCESS", "EVENT_BRAKE"]:
            for event in events:
                if event[0] == event_type:
                    return event[0], event[1]
            
        return None, None
    
    def get_sectors_until_next_private(self, all_path_sectors):
        needed_sectors = []
        num_curves = len(all_path_sectors)
        
        for i in range(num_curves):
            curve_idx = (self.current_curve_idx + i) % num_curves
            
            if curve_idx not in all_path_sectors:
                break
                
            sectors_on_curve = all_path_sectors.get(curve_idx, [])
            sorted_sectors = sorted(sectors_on_curve, key=lambda x: x.t_l)
            
            found_private = False
            for sector in sorted_sectors:
                if i == 0 and sector.t_u <= self.current_t:
                    continue
                    
                if i == 0 and sector.t_l <= self.current_t <= sector.t_u:
                    needed_sectors.append(sector)
                    continue
                
                if i > 0 or sector.t_l > self.current_t:
                    needed_sectors.append(sector)
            
            if sorted_sectors:
                last_sector = sorted_sectors[-1]
                if i == 0 and self.current_t > last_sector.t_u:
                    found_private = True
                elif i > 0 and last_sector.t_u < 0.99:
                    found_private = True
            else:
                found_private = True
            
            if found_private and i > 0:
                break
        
        return needed_sectors
    
    def get_upcoming_sectors(self, all_path_sectors, lookahead_curves=2):
        upcoming = []
        num_curves = len(all_path_sectors)
        
        for i in range(lookahead_curves):
            curve_idx = (self.current_curve_idx + i) % num_curves
            if curve_idx in all_path_sectors:
                for sector in all_path_sectors[curve_idx]:
                    if i == 0:
                        if sector.t_l > self.current_t:
                            upcoming.append((curve_idx, sector))
                    else:
                        upcoming.append((curve_idx, sector))
        
        return upcoming

    def in_private_sector(self, current_curve_sector):
        for sector in current_curve_sector:
            if sector.t_l <= self.current_t <= sector.t_u:
                return False
        return True
        
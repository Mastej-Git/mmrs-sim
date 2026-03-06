import numpy as np
from scipy.ndimage import label


class Sector:
    def __init__(self, t_l, t_u, resource_ids):    
        self.t_l = t_l
        self.t_u = t_u
        self.resource_ids = resource_ids
        self.is_private = len(resource_ids) == 0

        self.t_querry = None
        self.t_critical = None

    def __str__(self) -> str:
        return f"t_l: {self.t_l}, t_u: {self.t_u}, resource_id: {self.resource_ids}, is_private: {self.is_private}"

class CollisionSectorAlgorithm:

    def get_resource_key(self, agv1_id, curve1_idx, agv2_id, curve2_idx):
        pair = sorted([(agv1_id, curve1_idx), (agv2_id, curve2_idx)])
        return f"res_{pair[0][0]}_{pair[0][1]}_x_{pair[1][0]}_{pair[1][1]}"

    def get_bezier_aabb(self, verts, padding=0.0):
        p0, p1, p2 = map(np.array, verts)
        denom = p0 - 2*p1 + p2
        num = p0 - p1
        
        with np.errstate(divide='ignore', invalid='ignore'):
            t_ext = np.where(np.abs(denom) > 1e-12, num / denom, -1.0)
        
        candidates = [p0, p2]
        for i in range(2):
            if 0 < t_ext[i] < 1:
                t = t_ext[i]
                pt = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2
                candidates.append(pt)
                
        pts = np.array(candidates)
        return np.min(pts, axis=0) - padding, np.max(pts, axis=0) + padding
    
    def merge_sectors(self, sectors: list[Sector]) -> list[Sector]:
        if not sectors:
            return []

        sorted_s = sorted(sectors, key=lambda x: x.t_l)
        merged = []

        if sorted_s:
            current_sector = Sector(
                t_l=sorted_s[0].t_l,
                t_u=sorted_s[0].t_u,
                resource_ids=list(sorted_s[0].resource_ids)
            )

            for next_s in sorted_s[1:]:
                if next_s.t_l <= current_sector.t_u + 1e-9:
                    current_sector.t_u = max(current_sector.t_u, next_s.t_u)
                    for rid in next_s.resource_ids:
                        if rid not in current_sector.resource_ids:
                            current_sector.resource_ids.append(rid)
                else:
                    merged.append(current_sector)
                    current_sector = Sector(
                        t_l=next_s.t_l,
                        t_u=next_s.t_u,
                        resource_ids=list(next_s.resource_ids)
                    )
            
            merged.append(current_sector)
        return merged

    def find_roots_quartic(self, fixed_point: np.ndarray, verts_other, R: float):
        p0, p1, p2 = map(np.array, verts_other)
        A = p0 - 2*p1 + p2
        B = 2*(p1 - p0)
        D = p0 - fixed_point
        
        coeffs = [
            np.dot(A, A),
            2 * np.dot(A, B),
            2 * np.dot(A, D) + np.dot(B, B),
            2 * np.dot(B, D),
            np.dot(D, D) - R**2
        ]
        
        roots = np.roots(coeffs)
        real_roots = roots[np.isreal(roots)].real
        valid = [np.clip(r, 0.0, 1.0) for r in real_roots if -1e-5 <= r <= 1.00001]
        return sorted(list(set(np.round(valid, 8))))

    def get_closest_t_to_point(self, point: np.ndarray, verts: list[tuple[float, float]]) -> float:
        p0, p1, p2 = map(np.array, verts)

        A = p0 - 2*p1 + p2
        B = 2*(p1 - p0)
        C = p0 - point
        
        # P'(t) = 2At + B
        # Solve: (At^2 + Bt + C) . (2At + B) = 0
        # 2(A.A)t^3 + 3(A.B)t^2 + (2A.C + B.B)t + B.C = 0
        coeffs = [
            2 * np.dot(A, A),
            3 * np.dot(A, B),
            2 * np.dot(A, C) + np.dot(B, B),
            np.dot(B, C)
        ]
        roots = np.roots(coeffs)
        candidates = [0.0, 1.0]
        for r in roots:
            if np.isreal(r) and 0 < r < 1:
                candidates.append(r.real)
        
        best_t = 0.0
        min_dist_sq = float('inf')
        for t in candidates:
            pt = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2
            d_sq = np.sum((pt - point)**2)
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                best_t = t
        return best_t

    def expand_sector_around_minimum_fast(self, t_star: float, v_star: float, verts1, verts2, R: float):
        v1 = np.array(verts1)
        v2 = np.array(verts2)
        
        queue = [(1, t_star), (2, v_star)]
        checked = set()
        extremes = {1: [t_star, t_star], 2: [v_star, v_star]}
        R_sq = R * R

        idx = 0
        while idx < len(queue):
            curve_id, param = queue[idx]
            idx += 1
            
            key = (curve_id, round(param, 7))
            if key in checked: 
                continue
            checked.add(key)

            curr_verts = v1 if curve_id == 1 else v2
            other_verts = v2 if curve_id == 1 else v1
            other_id = 2 if curve_id == 1 else 1

            p_fixed = (1-param)**2 * curr_verts[0] + 2*(1-param)*param * curr_verts[1] + param**2 * curr_verts[2]
            
            roots = self.find_roots_quartic(p_fixed, other_verts, R)
            
            for r_other in roots:
                p_other = (1-r_other)**2 * other_verts[0] + 2*(1-r_other)*r_other * other_verts[1] + r_other**2 * other_verts[2]
                r_back = self.get_closest_t_to_point(p_other, curr_verts)
                
                p_back = (1-r_back)**2 * curr_verts[0] + 2*(1-r_back)*r_back * curr_verts[1] + r_back**2 * curr_verts[2]
                dist_sq = np.sum((p_other - p_back)**2)
                
                if dist_sq <= R_sq + 1e-7:
                    expanded = False
                    if r_other < extremes[other_id][0]: 
                        extremes[other_id][0] = r_other
                        expanded = True
                    if r_other > extremes[other_id][1]: 
                        extremes[other_id][1] = r_other
                        expanded = True
                    if r_back < extremes[curve_id][0]: 
                        extremes[curve_id][0] = r_back
                        expanded = True
                    if r_back > extremes[curve_id][1]: 
                        extremes[curve_id][1] = r_back
                        expanded = True
                    
                    if expanded:
                        queue.append((other_id, r_other))
                        queue.append((curve_id, r_back))

        return extremes[1][0], extremes[1][1], extremes[2][0], extremes[2][1]
    
    def process_curve_pair_multi(self, agv1_id, c1_idx, verts1, agv2_id, c2_idx, verts2, r1: float, r2: float, emergency_factor: float=1.1):
        R = (r1 + r2) * emergency_factor
        R_sq = R**2
        
        aabb1_min, aabb1_max = self.get_bezier_aabb(verts1, padding=R)
        aabb2_min, aabb2_max = self.get_bezier_aabb(verts2)
        if not (np.all(aabb1_max >= aabb2_min) and np.all(aabb2_max >= aabb1_min)):
            return [], []

        grid_n = 40
        t = np.linspace(0, 1, grid_n)
        v = np.linspace(0, 1, grid_n)
        T, V = np.meshgrid(t, v)
        
        p0_1, p1_1, p2_1 = map(np.array, verts1)
        p0_2, p1_2, p2_2 = map(np.array, verts2)
        
        t_flat = T.flatten()[:, None]
        v_flat = V.flatten()[:, None]
        
        pts1 = (1-t_flat)**2 * p0_1 + 2*(1-t_flat)*t_flat * p1_1 + t_flat**2 * p2_1
        pts2 = (1-v_flat)**2 * p0_2 + 2*(1-v_flat)*v_flat * p1_2 + v_flat**2 * p2_2
        
        dist_sq = np.sum((pts1 - pts2)**2, axis=1).reshape((grid_n, grid_n))

        mask = dist_sq < R_sq
        if not np.any(mask):
            return [], []

        labeled_array, num_features = label(mask)

        resource_id = self.get_resource_key(agv1_id, c1_idx, agv2_id, c2_idx)
        
        sectors1, sectors2 = [], []
        # pair_id = [verts1, verts2]

        for i in range(1, num_features + 1):
            cluster_mask = (labeled_array == i)
            masked_dist = np.where(cluster_mask, dist_sq, np.inf)
            idx = np.unravel_index(np.argmin(masked_dist), dist_sq.shape)
            
            t_star, v_star = T[idx], V[idx]

            tl, tu, vl, vu = self.expand_sector_around_minimum_fast(t_star, v_star, verts1, verts2, R)
            sectors1.append(Sector(tl, tu, [resource_id]))
            sectors2.append(Sector(vl, vu, [resource_id]))

        return self.merge_sectors(sectors1), self.merge_sectors(sectors2)


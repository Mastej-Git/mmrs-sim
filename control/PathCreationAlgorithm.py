import numpy as np
import matplotlib.patches as patches


class PathCreationAlgorithm:

    def __init__(self):
        pass

    def bezier_tangent(self, t: float, verts: list[tuple[int, int]]):
        p0, p1, p2 = map(np.array, verts)
        d = 2 * (1 - t) * (p1 - p0) + 2 * t * (p2 - p1)
        return float(d[0]), float(d[1])

    def _normalize_vec(self, vx: float, vy: float, length: float):
        norm = np.hypot(vx, vy)
        if norm == 0:
            return 0.0, length
        s = length / norm
        return vx * s, vy * s

    def create_path(self, marked_states: list[tuple[int, int]], orientation: tuple[int, int], radius: float) -> None:
        bezier_points = []

        start = 0
        end = np.array(marked_states[0])
        
        i = 0
        lap_ms_len = len(marked_states)
        
        while True:

            if i == len(marked_states) - 1:
                break

            if lap_ms_len == len(marked_states):
                start = end
                end = np.array(marked_states[i + 1])
            else:
                start = np.array(marked_states[i])
                end = np.array(marked_states[i + 1])
                lap_ms_len = len(marked_states)
                
            if i == 0:
                set_orient = orientation
            else:
                set_orient = np.array(self.bezier_tangent(1, bezier_points[i - 1]))

            ti_vec = set_orient
            pi_vec = end - start

            middle_point = start + radius * (ti_vec / np.linalg.norm(ti_vec))

            angle = np.arccos(np.dot(ti_vec, pi_vec)/(np.linalg.norm(ti_vec)*np.linalg.norm(pi_vec)))
            
            if angle < np.pi/2 and angle > -np.pi/2:
                tmp_list = [tuple(start.tolist()), tuple(middle_point.tolist()), tuple(end.tolist())]
                bezier_points.append(tmp_list)
            else:
                cross = ti_vec[0]*pi_vec[1] - ti_vec[1]*pi_vec[0]
                if cross > 0:
                    ti_vec = np.array([-ti_vec[1], ti_vec[0]])
                else:
                    ti_vec = np.array([ti_vec[1], -ti_vec[0]])
                additional_point = start + radius * 2 * (ti_vec / np.linalg.norm(ti_vec))
                tmp_list = [tuple(start.tolist()), tuple(middle_point.tolist()), tuple(additional_point.tolist())]
                marked_states.insert(i + 1, tuple(additional_point.tolist()))
                bezier_points.append(tmp_list)
            i += 1

        return bezier_points
    
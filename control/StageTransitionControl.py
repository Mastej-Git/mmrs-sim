from control.AGV import AGV
from control.PathCreationAlgorithm import PathCreationAlgorithm
from control.CollisionSectorAlgorithm import CollisionSectorAlgorithm

from itertools import combinations

from new_tmp_2912 import process_curve_pair_multi

class StageTransitionControl:

    def __init__(self):
        self.agvs = []
        self.col_sectors = []

        self.path_creator = PathCreationAlgorithm()
        self.col_det_alg = CollisionSectorAlgorithm()

    def create_paths(self) -> None:
        for agv in self.agvs:
            path = self.path_creator.create_path(agv.marked_states.copy(), agv.orientation, agv.radius)
            agv.path = path

    def detec_col_sectors(self):
        for agv1, agv2 in combinations(self.agvs, 2):
            for i in range(len(agv1.path)):
                for j in range(len(agv2.path)):

                    curveA = agv1.path[i]
                    curveB = agv2.path[j]

                    s1, s2 = process_curve_pair_multi(curveA, curveB, agv2.radius, agv1.radius, emergency_factor=1.1)
                    if len(s1) != 0 and len(s2) != 0:
                        self.col_sectors.append((s1, s2))

    def get_agvs_number(self) -> int:
        return len(self.agvs)
    
    def load_agvs(self, loaded_agvs: dict[str, AGV]) -> None:
        for agv in loaded_agvs.values():
            self.agvs.append(agv)

    def trigger_path_creation(self) -> None:
        self.create_paths()
    
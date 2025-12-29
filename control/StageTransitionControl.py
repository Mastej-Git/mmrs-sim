from control.AGV import AGV
from control.PathCreationAlgorithm import PathCreationAlgorithm
from control.CollisionSectorAlgorithm import CollisionSectorAlgorithm

from itertools import combinations

# agv1 = AGV(
#     marked_states=[(1, 1), (3, 6), (5, 2), (8, 7), (4, 7), (9, 12), (2, 13)],
#     radius=0.5,
#     color="#12700EFF",
#     path_color="#17D220",
# )

# agv2 = AGV(
#     marked_states=[(1, 7), (4, 10), (7, 7), (10, 4),  (13, 7)],
#     radius=0.5,
#     color="#12700EFF",
#     path_color="#17D220",
# )

# agv3 = AGV(
#     marked_states=[(13, 7), (10, 10), (7, 7), (4, 4), (1, 7)],
#     radius=0.5,
#     color="#330DCEFF",
#     path_color="#2F75CB",
# )

class StageTransitionControl:

    def __init__(self, robot):
        # self.agvs = [agv1]
        # self.agvs = [agv2, agv3]
        self.agvs = []
        self.col_sectors = []

        self.path_creator = PathCreationAlgorithm()
        self.col_det_alg = CollisionSectorAlgorithm()

    def create_paths(self) -> None:
        for agv in self.agvs:
            path = self.path_creator.create_path(agv.marked_states.copy(), agv.orientation, agv.radius)
            # print(path)
            agv.path = path

    def detec_col_sectors(self):
        for agv1, agv2 in combinations(self.agvs, 2):
            for i in range(len(agv1.path)):
                for j in range(len(agv2.path)):

                    curveA = agv1.path[i]
                    curveB = agv2.path[j]

                    sector_pair = self.col_det_alg.process_curve_pair_fast(curveA, curveB, agv2.radius, agv1.radius, ef=1.1)
                    if sector_pair is not None:
                        self.col_sectors.append(sector_pair)
                    # print("Sectors on curve A:")
                    # for sec in s1:
                    #     print(f"  [{sec.t_l:.6f}, {sec.t_u:.6f}] addresses={sec.addresses}")
                    # print("Sectors on curve B:")
                    # for sec in s2:
                    #     print(f"  [{sec.t_l:.6f}, {sec.t_u:.6f}] addresses={sec.addresses}")

    def get_agvs_number(self) -> int:
        return len(self.agvs)
    
    def load_agvs(self, loaded_agvs: dict[str, AGV]) -> None:
        for agv in loaded_agvs.values():
            self.agvs.append(agv)

    def trigger_path_creation(self) -> None:
        self.create_paths()
    
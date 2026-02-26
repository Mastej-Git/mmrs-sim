import yaml

from control.AGV import AGV 


class YamlAGVLoader:

    def __init__(self):
        self.file_name = "agvs_desc/test.yaml"
        # self.file_name = "agvs_desc/two_circles.yaml"
        # self.file_name = "agvs_desc/three_circles.yaml"
        # self.file_name = "agvs_desc/single_agv.yaml"

    def load_agvs_yaml(self):
        with open(self.file_name) as f:
            cfg = yaml.safe_load(f)

        agvs = {}
        for agv_cfg in cfg["agvs"]:
            agv = AGV(
                agv_id=int(agv_cfg["id"][-1]),
                marked_states=[tuple(p) for p in agv_cfg["marked_states"]],
                orientation=tuple(agv_cfg["orientation"]),
                radius=agv_cfg["radius"],
                max_v=float(agv_cfg["max_v"]),
                max_a=float(agv_cfg["max_a"]),
                color=agv_cfg["color"],
                path_color=agv_cfg["path_color"],
            )
            agvs[agv_cfg["id"]] = agv
            if "path" in agv_cfg:
                agv.path = [[tuple(coords) for coords in bc] for bc in agv_cfg["path"]]

        return agvs
import yaml

from control.AGV import AGV 


class YamlAGVLoader:

    def __init__(self):
        self.file_name = "agvs.yaml"

    def load_agvs_yaml(self):
        with open("agvs.yaml") as f:
            cfg = yaml.safe_load(f)

        agvs = {}
        for agv_cfg in cfg["agvs"]:
            agv = AGV(
                marked_states=[tuple(p) for p in agv_cfg["marked_states"]],
                orientation=tuple(agv_cfg["orientation"]),
                radius=agv_cfg["radius"],
                color=agv_cfg["color"],
                path_color=agv_cfg["path_color"],
            )
            agvs[agv_cfg["id"]] = agv

        return agvs
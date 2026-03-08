import numpy as np


class MapLoader:

    def __init__(self):
        self.file_name = "maps/Berlin_0_256.map"
        self.file_name = "maps/Berlin_0_1024.map"

    def load_map(self) -> np.ndarray:
        with open(self.file_name, 'r') as f:
            lines = f.readlines()
        
        height = int(lines[1].split()[1])
        width = int(lines[2].split()[1])
        
        map_lines = lines[4:4 + height]
        
        self.map_data = np.zeros((height, width), dtype=np.uint8)
        for row, line in enumerate(map_lines):
            for col, char in enumerate(line.rstrip('\n')):
                if col < width:
                    self.map_data[row, col] = 1 if char == '@' else 0
        
        return self.map_data
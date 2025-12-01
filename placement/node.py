import random
import numpy as np
from typing import Tuple, List

# Shared 
def get_valid_position_on_map(map_data, safe_range_x, safe_range_y, center=None, radius=None):
    min_x, max_x = safe_range_x
    min_y, max_y = safe_range_y
    w, h = map_data.shape
    
    current_radius = radius
    attempt_count = 0
    
    while True:
        attempt_count += 1
        
        if attempt_count > 100:
            center = None 
        
        if attempt_count > 2000:
            return (random.uniform(min_x, max_x), random.uniform(min_y, max_y))

        if center and current_radius:
            nx = center[0] + random.uniform(-current_radius, current_radius)
            ny = center[1] + random.uniform(-current_radius, current_radius)
            rx = max(min_x, min(max_x, nx))
            ry = max(min_y, min(max_y, ny))
        else:
            rx = random.uniform(min_x, max_x)
            ry = random.uniform(min_y, max_y)
        
        ix, iy = int(rx), int(ry)
        if 0 <= ix < w and 0 <= iy < h:
            # Using Env < 1e5 implies Reachable AND Not Obstacle
            if map_data[ix, iy] < 1e5:
                return (rx, ry)
        
        if center and current_radius:
            current_radius *= 1.1

class Metric:
    def __init__(self):
        self.cost_obs = 0.0
        self.cost_connect = 0.0
        self.cost_coll = 0.0
        self.cost_travel = 0.0
        self.cost_cover = 0.0
        self.cost_all = 0.0

    def reset(self):
        self.cost_obs = 0.0
        self.cost_connect = 0.0
        self.cost_coll = 0.0
        self.cost_travel = 0.0
        self.cost_cover = 0.0
        self.cost_all = 0.0

    def as_tuple(self):
        return (self.cost_obs, self.cost_connect, self.cost_coll, self.cost_travel, self.cost_cover, self.cost_all)


# PSO Particle
class Particle:
    def __init__(self, num_robots, map_data, safe_x, safe_y, start_center=None, start_radius=None):
        self.robots_number = num_robots
        self.map_data = map_data 
        self.safe_x = safe_x
        self.safe_y = safe_y
        
        # Init Position
        self.position = []
        for _ in range(self.robots_number):
            pos = get_valid_position_on_map(self.map_data, self.safe_x, self.safe_y, start_center, start_radius)
            self.position.append(pos)
            
        # Init Velocity
        self.velocity = [(0.0, 0.0) for _ in range(self.robots_number)]
        
        self.best_position = list(self.position)
        self.best_cost = float("inf")
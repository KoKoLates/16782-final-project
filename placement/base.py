import math
import numpy as np
from numba import jit
from typing import List, Tuple
from core import Env

@jit(nopython=True)
def jit_obstacle_block(x1, y1, x2, y2, map_data):
    dx = x2 - x1
    dy = y2 - y1
    dist = math.sqrt(dx*dx + dy*dy)
    num_samples = int(dist)
    
    if num_samples == 0: return False
    
    stepx = dx / dist
    stepy = dy / dist
    
    for k in range(1, num_samples):
        px = int(x1 + stepx * k)
        py = int(y1 + stepy * k)
        # Boundary Check
        if px < 0 or py < 0 or px >= map_data.shape[0] or py >= map_data.shape[1]:
            continue
        # Obstacle Check
        if map_data[px, py] >= 1e9:
            return True
    return False

@jit(nopython=True)
def jit_add_signal(signal_map, map_data, cx, cy, radius, gain):
    w, h = signal_map.shape
    x_min = max(0, int(cx - radius))
    x_max = min(w, int(cx + radius + 1))
    y_min = max(0, int(cy - radius))
    y_max = min(h, int(cy + radius + 1))
    
    if x_max <= x_min or y_max <= y_min: return

    sigma2 = 2.0 * (radius / 2.0)**2
    
    for i in range(x_min, x_max):
        for j in range(y_min, y_max):
            dist_sq = (i - cx)**2 + (j - cy)**2
            if dist_sq <= radius**2:
                val = gain * math.exp(-dist_sq / sigma2)
                # Check Line-of-Sight
                if jit_obstacle_block(cx, cy, i, j, map_data):
                    val *= 0.5
                signal_map[i, j] += val

@jit(nopython=True)
def jit_calculate_coverage(map_data, positions, connected, tower_pos, 
                           tower_gain, robot_gain, signal_max, signal_threshold):
    w, h = map_data.shape
    signal_map = np.zeros((w, h), dtype=np.float32)
    
    # 1. Tower Signal
    jit_add_signal(signal_map, map_data, tower_pos[0], tower_pos[1], tower_gain, 10.0)
    
    # 2. Robot Signals (Only connected ones)
    for k in range(len(positions)):
        if connected[k]:
            jit_add_signal(signal_map, map_data, positions[k][0], positions[k][1], robot_gain, 10.0)
            
    # 3. Count Uncovered
    covered_count = 0
    for i in range(w):
        for j in range(h):
            # Ignore Obstacles (>= 1e9) for coverage counting
            if map_data[i, j] >= 1e9:
                signal_map[i, j] = 0.0 
            else:
                if signal_map[i, j] > signal_max: signal_map[i, j] = signal_max
                if signal_map[i, j] >= signal_threshold: covered_count += 1
                    
    return (w * h) - covered_count

@jit(nopython=True)
def jit_calculate_travel_cost(positions, map_data):
    total_dist = 0.0
    penalty = 0.0
    
    for i in range(len(positions)):
        x = int(positions[i][0])
        y = int(positions[i][1])
        
        # Boundary Safe Check
        if x < 0 or x >= map_data.shape[0] or y < 0 or y >= map_data.shape[1]:
            penalty += 1e9
            continue

        cost = map_data[x, y]
        
        # Check Unreachable / Obstacle
        if cost >= 1e5:
            penalty += 1e6 # penalty for unreachable areas
        else:
            total_dist += cost
            
    return total_dist + penalty

def calculate_physics_factor(threshold):
    try:
        t = max(1e-3, threshold)
        term = 20.0 / t
        if term <= 1.0: return 0.0
        return math.sqrt(0.5 * math.log(term))
    except ValueError:
        return 1.0

class Coverage:
    def __init__(self, env: Env):
        self.env = env
        self.w, self.h = env.shape
        self.num_robots = env.robots_number
        self.map_data = np.array(self.env.map, dtype=np.float32)
        self.tower_pos = (float(self.w // 2), float(self.h // 2))

    def process(self) -> List[Tuple[int, int]]:
        raise NotImplementedError
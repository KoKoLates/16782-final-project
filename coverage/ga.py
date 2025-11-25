import numpy as np
import random
import math
import time
from core import Env
from typing import List, Tuple
from dataclasses import dataclass
from numba import jit


@jit(nopython=True)
def obstacle_block(x1, y1, x2, y2, map_data):
    dx = x2 - x1
    dy = y2 - y1
    dist = math.sqrt(dx*dx + dy*dy)
    num_samples = int(dist)
    
    if num_samples == 0: 
        return False
    
    stepx = dx / dist
    stepy = dy / dist
    
    for k in range(1, num_samples):
        px = int(x1 + stepx * k)
        py = int(y1 + stepy * k)
        if px < 0 or py < 0 or px >= map_data.shape[0] or py >= map_data.shape[1]:
            continue

        if map_data[px, py] == 1:
            return True
            
    return False

@jit(nopython=True)
def lap_signal(signal_map, map_data, cx, cy, radius, gain):
    w, h = signal_map.shape
    
    x_min = max(0, int(cx - radius))
    x_max = min(w, int(cx + radius + 1))
    y_min = max(0, int(cy - radius))
    y_max = min(h, int(cy + radius + 1))
    
    if x_max <= x_min or y_max <= y_min:
        return

    sigma2 = 2.0 * (radius / 2.0)**2
    
    for i in range(x_min, x_max):
        for j in range(y_min, y_max):
            dist_sq = (i - cx)**2 + (j - cy)**2
            
            if dist_sq <= radius**2:
                val = gain * math.exp(-dist_sq / sigma2)
                
                if obstacle_block(cx, cy, i, j, map_data):
                    val *= 0.5
                
                signal_map[i, j] += val

@jit(nopython=True)
def cal_coverage(map_data, positions, connected, tower_pos, 
                           tower_gain, robot_gain, signal_max, signal_threshold):
    w, h = map_data.shape
    signal_map = np.zeros((w, h), dtype=np.float32)
    
    lap_signal(signal_map, map_data, tower_pos[0], tower_pos[1], tower_gain, 10.0)
    
    for k in range(len(positions)):
        if connected[k]:
            rx = positions[k][0]
            ry = positions[k][1]
            lap_signal(signal_map, map_data, rx, ry, robot_gain, 10.0)
            
    covered_count = 0
    for i in range(w):
        for j in range(h):
            if map_data[i, j] == 1:
                signal_map[i, j] = 0.0
            else:
                if signal_map[i, j] > signal_max:
                    signal_map[i, j] = signal_max
                
                if signal_map[i, j] >= signal_threshold:
                    covered_count += 1
                    
    return (w * h) - covered_count

@dataclass
class GAParams:
    pop_size: int = 50          
    generations: int = 500 
    mutation: float = 0.1 

    # Signal 
    tower_gain: int = 10
    robot_gain: int = 4
    signal_threshold: float = 1.0
    signal_max: float = 10.0
    
    # Cost (align with PSO)
    w_obstacle: float = 1e9      
    w_hard_constraint: float = 1e7 
    w_connect: float = 1e5       
    
    # Init Strategy
    init_method: str = "connected" 
    
    # Early Stop
    patience: int = 10 
    early_stop: float = 1.0

class GeneticAlgorithm:
    def __init__(self, env: Env, params: GAParams = None):
        self.env = env
        self.num_robots = env.robots_number
        self.w, self.h = env.shape
        self.params = params if params else GAParams()
        
        if hasattr(self.env, 'map'):
            self.map_data = np.array(self.env.map, dtype=np.int32)
        else:
            self.map_data = np.zeros((self.w, self.h), dtype=np.int32)
            for x in range(self.w):
                for y in range(self.h):
                    if self.env.is_obstacle(x, y):
                        self.map_data[x, y] = 1
        
        margin = int(self.params.robot_gain)
        self.safe_x_range = (margin, self.w - margin - 1)
        self.safe_y_range = (margin, self.h - margin - 1)
        self.tower_pos = (float(self.w // 2), float(self.h // 2)) 

        try:
            threshold = max(1e-3, self.params.signal_threshold)
            term = 14.0 / threshold 
            if term <= 1.0:
                physics_factor = 0.0
            else:
                physics_factor = math.sqrt(0.5 * math.log(term))
        except ValueError:
            physics_factor = 1.0

        self.connect_factor = min(1.0, physics_factor)
    
        tr_dist = (self.params.tower_gain + self.params.robot_gain) * self.connect_factor
        self.tower_range_sq = tr_dist ** 2
        
        rr_dist = (self.params.robot_gain + self.params.robot_gain) * self.connect_factor
        self.robot_range_sq = rr_dist ** 2

    def get_valid_position(self, center: Tuple[int, int] = None, radius: int = None) -> Tuple[int, int]:
        min_x, max_x = self.safe_x_range
        min_y, max_y = self.safe_y_range
        
        attempts = 20 
        
        for _ in range(attempts):
            if center and radius:
                cx, cy = center
                nx = cx + random.randint(-radius, radius)
                ny = cy + random.randint(-radius, radius)
                rx = max(min_x, min(max_x, nx))
                ry = max(min_y, min(max_y, ny))
            else:
                rx = random.randint(min_x, max_x)
                ry = random.randint(min_y, max_y)

            if self.map_data[rx, ry] == 0:
                return (rx, ry)
        
        return (random.randint(min_x, max_x), random.randint(min_y, max_y))

    def cobstacle(self, positions):
        cost = 0
        min_x, max_x = self.safe_x_range
        min_y, max_y = self.safe_y_range

        for (x, y) in positions:
            if not (min_x <= x <= max_x and min_y <= y <= max_y):
                 cost += self.params.w_obstacle
                 continue
            
            ix, iy = int(x), int(y)
            if self.map_data[ix, iy] == 1:
                cost += self.params.w_obstacle

        return cost
    
    def cconnectivity(self, positions):
        n = len(positions)
        connected = [False] * n
        queue = []
        
        tx, ty = self.tower_pos
        
        for i, (px, py) in enumerate(positions):
            dist_sq = (px - tx)**2 + (py - ty)**2
            if dist_sq < self.tower_range_sq:
                connected[i] = True
                queue.append(i)
        
        if not queue:
            dist_penalty = 0
            for px, py in positions:
                dist_penalty += math.sqrt((px - tx)**2 + (py - ty)**2) * 100
            return self.params.w_hard_constraint + dist_penalty, connected

        head = 0
        while head < len(queue):
            curr_idx = queue[head]
            head += 1
            
            cx, cy = positions[curr_idx]
            
            for j in range(n):
                if not connected[j]:
                    jx, jy = positions[j]
                    dist_sq = (cx - jx)**2 + (cy - jy)**2
                    if dist_sq < self.robot_range_sq:
                        connected[j] = True
                        queue.append(j)

        if all(connected):
            return 0, connected
            
        cost_connect = 0
        connected_indices = [i for i, c in enumerate(connected) if c]
        disconnected_indices = [i for i, c in enumerate(connected) if not c]
        
        for i in disconnected_indices:
            ix, iy = positions[i]
            min_dist_sq = (ix - tx)**2 + (iy - ty)**2
            
            for j in connected_indices:
                jx, jy = positions[j]
                d_sq = (ix - jx)**2 + (iy - jy)**2
                if d_sq < min_dist_sq:
                    min_dist_sq = d_sq
            
            cost_connect += self.params.w_connect + (math.sqrt(min_dist_sq) * 10)

        return cost_connect, connected

    def ccoverage(self, positions, connected):
        pos_arr = np.array(positions, dtype=np.float64)
        conn_arr = np.array(connected, dtype=np.bool_)
        
        return cal_coverage(
            self.map_data, 
            pos_arr, 
            conn_arr, 
            self.tower_pos,
            float(self.params.tower_gain), 
            float(self.params.robot_gain), 
            float(self.params.signal_max), 
            float(self.params.signal_threshold)
        )

    def cal_fitness(self, positions: List[Tuple[int, int]]) -> float:
        float_positions = [(float(x), float(y)) for x, y in positions]
        
        cost_obs = self.cobstacle(float_positions)
        if cost_obs > 0:
            return cost_obs + self.params.w_hard_constraint
            
        cost_connect, connected = self.cconnectivity(float_positions)
        if cost_connect > 0:
             return cost_obs + cost_connect + self.params.w_hard_constraint

        cost_cover = self.ccoverage(float_positions, connected)
        
        return cost_obs + cost_connect + cost_cover

    def process(self) -> List[Tuple[int, int]]:
        population = self.init_population()
        chosen_solution = None
        chosen_cost = float('inf') 
        
        stall_count = 0
        history_best_cost = float('inf')

        for gen in range(self.params.generations):
            costs = [self.cal_fitness(ind) for ind in population]
            
            current_min_cost = min(costs)
            if current_min_cost < chosen_cost:
                chosen_cost = current_min_cost
                chosen_solution = population[costs.index(current_min_cost)]
            
            if (history_best_cost - chosen_cost) > self.params.early_stop:
                stall_count = 0
                history_best_cost = chosen_cost
            else:
                stall_count += 1
                
            if stall_count >= self.params.patience:
                print(f"Early stop at Gen {gen+1}, Best Cost: {chosen_cost:.2f}")
                break

            new_population = [chosen_solution] 
            while len(new_population) < self.params.pop_size:
                p1 = self.parent_select(population, costs)
                p2 = self.parent_select(population, costs)
                c1, c2 = self.crossover(p1, p2)
                self.mutate(c1)
                self.mutate(c2)
                new_population.extend([c1, c2])
            
            population = new_population[:self.params.pop_size]

        print(f"End. Best Cost: {chosen_cost:.2f}")
        return chosen_solution if chosen_solution else population[0]

    def init_population(self) -> List[List[Tuple[int, int]]]:
        pop = []
        is_connected_init = (self.params.init_method == "connected")
        factor = self.connect_factor if is_connected_init else 1.0
        
        if is_connected_init:
            init_radius = int(self.params.tower_gain * 2 * factor) 
            center = (int(self.tower_pos[0]), int(self.tower_pos[1]))  
        else:
            init_radius = None
            center = None

        for _ in range(self.params.pop_size):
            individual = []
            for _ in range(self.num_robots):
                pos = self.get_valid_position(center, init_radius)
                individual.append(pos)
            pop.append(individual)
        return pop

    def mutate(self, ind):
        if random.random() < self.params.mutation:
            idx = random.randint(0, len(ind) - 1)
            if random.random() < 0.5:
                # Local 
                ind[idx] = self.get_valid_position(center=ind[idx], radius=2)
            else:
                # Global 
                ind[idx] = self.get_valid_position()

    def parent_select(self, pop, scores, k=3):
        indices = random.sample(range(len(pop)), k)
        best_idx = min(indices, key=lambda i: scores[i])

        return pop[best_idx]
    
    def crossover(self, p1, p2):
        if len(p1) < 2: 
            return list(p1), list(p2)
        pt = random.randint(1, len(p1) - 1)

        return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
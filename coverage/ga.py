import numpy as np
import random
import math
from core import Env
from typing import List, Tuple  
from dataclasses import dataclass


# ga parameter setting
@dataclass
class GAParams:
    pop_size: int = 50          # Pop size
    generations: int = 100      # Iterations
    mutation: float = 0.1 

    signal_range: float = 20.0  # Signal radius
    gain: float = 1.0           # Maximum signal strength
    threshold: float = 0.05     # Threshold for valid connectivity

    connectivity_reward: float = 1000.0 # Bonus for picking up lost robot
    
    @property
    def sigma(self) -> float:
        return self.signal_range / 2.0

class GeneticAlgorithm:
    def __init__(self, env: Env, params: GAParams = None):
        self.env = env
        self.num_robots = env.robots_number
        self.w, self.h = env.shape
        
        if params is None:
            self.params = GAParams()
        else:
            self.params = params
        
        # stamped heatmap (to reduce computation)
        self.signal_stamped, self.stamped_radius = self.precompute_signal_stamped()
        # precompute obstacle mask
        self.static_mask = np.ones((self.w, self.h))
        for x in range(self.w):
            for y in range(self.h):
                if self.env.is_obstacle(x, y):
                    self.static_mask[x, y] = 0.0

    def process(self) -> List[Tuple[int, int]]:
        # initialize 
        population = self.init_population()
        chosen_solution = None
        chosen_fitness = -float('inf')

        # evolution (loop)
        for gen in range(self.params.generations):
            # cal fitneess
            fitness_scores = [self.calculate_fitness(ind) for ind in population]
            # select
            current_best_fit = max(fitness_scores)
            if current_best_fit > chosen_fitness:
                chosen_fitness = current_best_fit
                best_idx = fitness_scores.index(current_best_fit)
                chosen_solution = population[best_idx]
                
            # reproduction
            new_population = [chosen_solution]
            while len(new_population) < self.params.pop_size:
                # Selection (random to avoid local min)
                p1 = self.parent_select(population, fitness_scores)
                p2 = self.parent_select(population, fitness_scores)
                
                # Crossover
                c1, c2 = self.crossover(p1, p2)
                
                # Mutation
                self.mutate(c1)
                self.mutate(c2)
                
                new_population.extend([c1, c2])
            
            # Trim 
            population = new_population[:self.params.pop_size]

        # print(f"GA: Evolution complete. Best Fitness: {chosen_fitness:.2f}")
        
        if chosen_solution is None:
            return population[0]
            
        return chosen_solution
        
    def precompute_signal_stamped(self) -> Tuple[np.ndarray, int]:
        r = int(self.params.sigma * 3) 
        x = np.arange(2*r+1) - r
        y = np.arange(2*r+1) - r
        X, Y = np.meshgrid(x, y)
        dist = X**2 + Y**2
        temp = self.gaussian_signal(dist)
        # delete noise
        temp[temp < 0.01] = 0.0
        return temp, r

    # generate the initial population in random 
    def init_population(self) -> List[List[Tuple[int, int]]]:
        pop = []
        for _ in range(self.params.pop_size):
            individual = []
            for _ in range(self.num_robots):
                while True:
                    rx = random.randint(0, self.w - 1)
                    ry = random.randint(0, self.h - 1)
                    if not self.env.is_obstacle(rx, ry):
                        individual.append((rx, ry))
                        break
            pop.append(individual)
        return pop
    
    def gaussian_signal(self, dist_sq: float) -> float:
        # cal strength -> Gaussian decay
        return self.params.gain * np.exp(-dist_sq / (2 * self.params.sigma**2))
    
    def is_blocked(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
        x1, y1 = p1
        x2, y2 = p2
        dist = math.hypot(x2 - x1, y2 - y1)
        
        if dist < 1.0: return False 
        
        # check " " times per block
        density = 5  
        steps = int(dist * density)

        for i in range(1, steps):
            t = i / steps
            xt = int(x1 + (x2 - x1) * t)
            yt = int(y1 + (y2 - y1) * t)
            if self.env.is_obstacle(xt, yt):
                return True
        return False

    def get_connect_list(self, individual: List[Tuple[int, int]]) -> List[int]:

        # builds the connectivity gang and returns the indices 
        n = len(individual)
        if n == 0: return []

        # build connect net
        net = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = (individual[i][0]-individual[j][0])**2 + (individual[i][1]-individual[j][1])**2
                sig = self.gaussian_signal(dist_sq)

                # weaken signal if blocked
                if self.is_blocked(individual[i], individual[j]):
                    sig *= 0.5
                
                # check threshold
                if sig >= self.params.threshold:
                    net[i].append(j)
                    net[j].append(i)

        # find largest connectitivity using BFS 
        visited = set()
        components = []
        for i in range(n):
            if i not in visited:
                queue = [i]
                component = []
                visited.add(i)
                while queue:
                    node = queue.pop(0)
                    component.append(node)
                    for neighbor in net[node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                components.append(component)
        if not components:
            return[]
        return max(components, key=len)    
    
    def calculate_fitness(self, individual: List[Tuple[int, int]]) -> float:
        """
        (not sure)
        1. Find Largest Connected Component
        2. Fitness = (Coverage of the largest connectivity) + (Size * Reward)
        """
        # identyfy connected robots (re: list index)
        connected_indices = self.get_connect_list(individual)
        num_connected = len(connected_indices)
        
        if num_connected == 0:
            return 0.0
        
        bonus = num_connected * self.params.connectivity_reward

        # coverage cal
        signal_map = np.zeros((self.w, self.h))
        r = self.stamped_radius
        h, w = self.h, self.w

        # only consider connected robots
        for idx in connected_indices:
            rx, ry = individual[idx]
            
            # Calculate bounding box on the map
            x0 = max(0, rx - r)
            x1 = min(w, rx + r + 1)
            y0 = max(0, ry - r)
            y1 = min(h, ry + r + 1)
            
            # Calculate corresponding slice on the template
            tx0 = max(0, r - rx)
            tx1 = tx0 + (x1 - x0)
            ty0 = max(0, r - ry)
            ty1 = ty0 + (y1 - y0)
            
            # Matrix Addition (Superposition)
            signal_map[x0:x1, y0:y1] += self.signal_stamped[tx0:tx1, ty0:ty1]

        # Apply static mask to remove signal inside walls
        signal_map *= self.static_mask
        
        # Clip max signal (optional, simulates saturation)
        signal_map = np.clip(signal_map, 0, self.params.gain)

        coverage_score = float(np.sum(signal_map))

        return coverage_score + bonus

    def parent_select(self, pop, scores, k=3):
        indices = random.sample(range(len(pop)), k)
        best_idx = max(indices, key=lambda i: scores[i])
        return pop[best_idx]
    
    def crossover(self, p1, p2):
        if len(p1) < 2: return list(p1), list(p2)
        pt = random.randint(1, len(p1) - 1)
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
        return c1, c2

    def mutate(self, ind):
        if random.random() < self.params.mutation:
            idx = random.randint(0, len(ind) - 1)
            for _ in range(10):
                nx = random.randint(0, self.w - 1)
                ny = random.randint(0, self.h - 1)
                if not self.env.is_obstacle(nx, ny):
                    ind[idx] = (nx, ny)
                    break

# 有疊加要怎加？(try to lower the coomplexity)
'''
先計算自身的 看有沒有過threshold
沒有在try看看疊加隔壁機器人嘛
'''

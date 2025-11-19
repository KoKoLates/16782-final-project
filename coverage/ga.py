import numpy as np
import random
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass
from core.env import Env

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
        
    def process(self) -> List[Tuple[int, int]]:
        # initialize 
        population = self.init_population()
        chosen_solution = None
        chosen_fitness = -float('inf')

        # evolution (loop)
        for gen in range(self.params.generations):
            # cal fitneess
            fitness_scores = [self._calculate_fitness(ind) for ind in population]
            # select
            current_best_fit = max(fitness_scores)
            if current_best_fit > chosen_fitness:
                chosen_fitness = current_best_fit
                best_idx = fitness_scores.index(current_best_fit)
                chosen_solution = population[best_idx]
                print(f"  Gen {gen}: Best Fit = {chosen_fitness:.2f}")
        
        # reproduction
        new_population = [chosen_solution]
        ...
        


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
        
        # Sample density: check every 0.5 units
        steps = int(dist * 2) 
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

                # Strict Line-of-Sight check for inter-robot connection
                if self.is_blocked(individual[i], individual[j]):
                    sig *= 0.5
                
                # here 
        ...


    def calculate_fitness(self, individual: List[Tuple[int, int]]) -> float:
        """
        (not sure)
        1. Find Largest Connected Component
        2. Fitness = (Coverage) + (Size * Reward)
        """
        # identyfy connected robots (re: list index)
        connected_indices = self.get_connect_list(individual)
        num_connected = len(connected_indices)
        ...
    

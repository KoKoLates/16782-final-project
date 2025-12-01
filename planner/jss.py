import math
import heapq
from typing import List, Tuple, Dict, Optional, Set

from core import Env
from .node import State, Path
from .base import Planner, DIRECTIONS, COST_STRAIGHT, COST_DIAGONAL

# A joint configuration at a given time is just a tuple of positions:
# ((x0, y0), (x1, y1), ..., (x_{N-1}, y_{N-1}))
JointPos = Tuple[Tuple[int, int], ...]

class JointAStarPlanner(Planner):
    """
    Joint state-space A* planner.
    Plans for all robots simultaneously in the Cartesian product of their state spaces.
    """

    def __init__(self, env: Env):
        super().__init__(env)

    @staticmethod
    def joint_heuristic(joint_pos: JointPos, goals: JointPos) -> float:
        """
        heuristics = sum of all robots' individual heuristics to their goals
        """
        h = 0.0
        for (x, y), (gx, gy) in zip(joint_pos, goals):
            h += Planner.heuristic((x, y), (gx, gy))
        return h

    def generate_joint_successors(
        self,
        current: JointPos,
        t: int,
        max_time: int
    ) -> List[JointPos]:
        """
        Generate all valid joint successors from 'current' at time t.

        Enforces:
          - each robot moves with DIRECTIONS (8-connect + wait)
          - robot must stay within bounds & avoid obstacles
          - diagonal moves can not go through obstacles
          - no two robots occupy the same cell (vertex collision)
          - no two robots swap positions (edge collision)
        """
        if t >= max_time:
            return []

        num_agents = len(current)
        env = self.env

        # For each agent, determine its possible next positions
        agent_next_positions: List[List[Tuple[int, int]]] = []

        for i in range(num_agents):
            cx, cy = current[i]
            options: List[Tuple[int, int]] = []

            for dx, dy in DIRECTIONS:
                nx, ny = cx + dx, cy + dy

                # Check map bounds & obstacles
                if not Planner.is_valid_location(nx, ny, env):
                    continue

                # Diagonal corner cutting check
                if dx != 0 and dy != 0:
                    if Planner.check_corner_cutting((cx, cy), (nx, ny), env):
                        continue

                options.append((nx, ny))

            # If no moves available, allow to stay
            if not options:
                if Planner.is_valid_location(cx, cy, env):
                    options.append((cx, cy))

            agent_next_positions.append(options)

        successors: List[JointPos] = []

        def backtrack(agent_idx: int, partial: List[Tuple[int, int]]):
            
            if agent_idx == num_agents:
                # next move for all robots decided
                next_joint = tuple(partial)

                # Vertex collision check
                if len(set(next_joint)) != num_agents:
                    return

                # Edge collision check
                for i in range(num_agents):
                    for j in range(i + 1, num_agents):
                        if (current[i] == next_joint[j] and
                                current[j] == next_joint[i]):
                            return

                successors.append(next_joint)
                return

            for pos in agent_next_positions[agent_idx]:
                partial.append(pos)
                backtrack(agent_idx + 1, partial)
                # pop to get new next move for current agent and run backtrack to 
                # get all the moves for the rest of the agents using recursive idea
                partial.pop()

        backtrack(0, [])
        return successors

    def joint_a_star(
        self,
        starts: JointPos,
        goals: JointPos,
        max_time: int = 100
    ) -> Optional[List[JointPos]]:
        """
        starts, goals: one (x, y) per robot, same ordering.
        Returns a list of joint positions from t=0...T if successful,
        otherwise None.
        """

        if len(starts) != len(goals):
            raise ValueError("Number of starts and goals must be the same.")

        # State in the search = (joint_pos, t)
        start_joint: JointPos = tuple(starts)
        start_key = (start_joint, 0)
        # priority queue (min-heap) sorted by f and g
        open_heap: List[Tuple[float, float, int, JointPos, int]] = []
        # [key,value], find the parent joint configure of key
        came_from: Dict[Tuple[JointPos, int], Tuple[JointPos, int]] = {}
        # g_score: Dict[(joint_pos,t) -> best_g]
        g_score: Dict[Tuple[JointPos, int], float] = {start_key: 0.0}

        start_h = self.joint_heuristic(start_joint, tuple(goals))
        counter = 0  # tie-breaker

        # Heap item: (f, g, tie_breaker, joint_pos, t)
        heapq.heappush(open_heap, (start_h, 0.0, counter, start_joint, 0))

        closed: Set[Tuple[JointPos, int]] = set()

        while open_heap:
            f, g, _, current_joint, t = heapq.heappop(open_heap)
            current_key = (current_joint, t)

            if current_key in closed:
                continue
            closed.add(current_key)

            # all robots at goals
            if current_joint == tuple(goals):
                # Reconstruct path backward using (joint_pos, t) keys
                path_joint: List[JointPos] = [current_joint]
                key = current_key
                while key in came_from:
                    key = came_from[key]
                    path_joint.append(key[0])
                path_joint.reverse()
                return path_joint

            # Expand successors
            successors = self.generate_joint_successors(current_joint, t, max_time)

            # enumerate all the successors
            for succ_joint in successors:
                succ_t = t + 1
                succ_key = (succ_joint, succ_t)

                # Compute the step cost: sum of per-agent movement costs
                step_cost = 0.0
                for (cx, cy), (nx, ny) in zip(current_joint, succ_joint):
                    dx = nx - cx
                    dy = ny - cy
                    if dx == 0 and dy == 0:
                        step_cost += 1.0 # wait cost = 1
                    elif dx != 0 and dy != 0:
                        step_cost += COST_DIAGONAL
                    else:
                        step_cost += COST_STRAIGHT

                tentative_g = g + step_cost

                if succ_key in g_score and tentative_g >= g_score[succ_key]:
                    continue  # not a better path

                g_score[succ_key] = tentative_g
                came_from[succ_key] = current_key
                h = self.joint_heuristic(succ_joint, tuple(goals))

                counter += 1
                heapq.heappush(
                    open_heap,
                    (tentative_g + h, tentative_g, counter, succ_joint, succ_t)
                )

        # No solution
        return None

    # public API
    def process(self, goals: List[Tuple[int, int]]) -> List[Path]:
        """
        goals: one goal (x, y) per robot, same order as starts.
        Returns:
          - List[Path], one Path per robot, where Path is a List[State].
          - Empty list if no solution.
        """
        num_agents = self.env.robots_number

        if len(goals) != num_agents:
            raise ValueError(
                f"Expected {num_agents} goals (one per robot), got {len(goals)}."
            )

        # Determine start positions.
        # allow for self-determined starts in env
        if hasattr(self.env, "starts") and len(self.env.starts) == num_agents:
            starts = list(self.env.starts)
        else:
            # default start positions from env, set in the mid of the map
            starts = list(self.starts)

        # Run joint A*
        joint_path = self.joint_a_star(tuple(starts), tuple(goals), max_time=100)
        if joint_path is None or len(joint_path) == 0:
            return []

        # create an empty path for each robot
        paths: List[Path] = [[] for _ in range(num_agents)]

        for t, joint in enumerate(joint_path):
            for i in range(num_agents):
                x, y = joint[i]
                paths[i].append(State(t=t, x=x, y=y))

        return paths

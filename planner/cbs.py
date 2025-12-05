# planner/cbs.py

import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Dict, Optional

from core import Env
from .node import State, Path, Constraint
from .base import Planner


@dataclass(order=True)
class CTNode:
    # Cost will be used as the priority in the heap
    cost: float
    constraints: List[Constraint] = field(compare=False, default_factory=list)
    paths: List[Path] = field(compare=False, default_factory=list)


@dataclass
class Conflict:
    robot_i: int
    robot_j: int
    time: int
    x: int
    y: int
    is_swap: bool
    # For swap conflicts we store the positions before/after
    pos_i_before: Tuple[int, int]
    pos_j_before: Tuple[int, int]
    pos_i_after: Tuple[int, int]
    pos_j_after: Tuple[int, int]


class CBSPlanner(Planner):
    """
    Conflict-Based Search planner.
    High-level search over a constraint tree (CT), low-level search is A*
    """

    def __init__(self, env: Env, max_time: int = 100):
        super().__init__(env)
        self.max_time = max_time
        self.goals: List[Tuple[int, int]] = []


    def process(self, goals: List[Tuple[int, int]]) -> List[Path]:
        """
        Main CBS planning method.
        low level CBS A* to get initial paths for all robots
        high level CBS loop to resolve conflicts
        """
        self.goals = goals
        num_robots = self.env.robots_number

        # --- 1. Low level CBS A* ---
        # initialize root CT node with empty constraints sets, and path for each robot computed by A*
        root_constraints: List[Constraint] = []
        root_paths: List[Path] = []
        for robot_id in range(num_robots):
            start_pos = self.starts[robot_id]
            goal_pos = self.goals[robot_id]

            constraint_states = self._collect_constraint_states(root_constraints, robot_id)

            path = self.state_time_a_star(
                start_pos=start_pos,
                goal_pos=goal_pos,
                constraints_list=constraint_states,
                env=self.env,
                check_collisions=True,
                max_time=self.max_time,
            )
            if path is None:
                # No path found
                return []

            root_paths.append(path)

        root_cost = Planner.calculate_total_path_cost(root_paths)
        root = CTNode(cost=root_cost, constraints=root_constraints, paths=root_paths)

        # --- 2. High-level CBS loop over CT nodes ---
        open_heap: List[CTNode] = []
        heapq.heappush(open_heap, root)
        while open_heap:
            node = heapq.heappop(open_heap)
            # find earliest conflict for the existing robot paths
            conflict = self._find_conflict(node.paths)
            # print(conflict)
            if conflict is None:
                # Conflict-free solution found
                # print("solution found")
                return node.paths

            # Expand this node into children, each adding one constraint
            children = self._expand_node(node, conflict)
            #print("node expanded")
            for child in children:
                if child is not None:
                    #print(child.paths)
                    heapq.heappush(open_heap, child)

        # No solution found
        return []


    @staticmethod
    def _get_state_at_time(path: Path, t: int) -> State:
        """
        Return the robot's state at time t.
        If t is beyond the end of the path, the robot waits at its last position.
        """
        if not path:
            # Should not really happen, but guard anyway
            return State(t=0, x=0, y=0)

        if t < len(path):
            return path[t]

        last = path[-1]
        # Stay in place but time keeps increasing
        return State(t=t, x=last.x, y=last.y)

    def _find_conflict(self, paths: List[Path]) -> Optional[Conflict]:
        """
        Find the earliest conflict between any pair of robots.
        Conflicts considered:
        - Vertex: same (x, y) at the same time t.
        - Swap: they swap positions between t and t+1.
        """
        num_robots = len(paths)
        if num_robots <= 1:
            return None

        max_len = max(len(p) for p in paths)

        # Check time steps 0..max_len (swap needs t and t+1)
        for t in range(max_len):
            if (t == 0) :
                continue
            for i in range(num_robots):
                for j in range(i + 1, num_robots):
                    # get state info for robots i and j at time t
                    si_t = self._get_state_at_time(paths[i], t)
                    sj_t = self._get_state_at_time(paths[j], t)

                    # Vertex conflict
                    if si_t.x == self.env.w // 2 and sj_t.x == self.env.w // 2 and si_t.y == self.env.h // 2 and sj_t.y == self.env.h // 2:
                        continue
                    if si_t.x == sj_t.x and si_t.y == sj_t.y:
                        return Conflict(
                            robot_i=i,
                            robot_j=j,
                            time=t,
                            x=si_t.x,
                            y=si_t.y,
                            is_swap=False,
                            pos_i_before=(si_t.x, si_t.y),
                            pos_j_before=(sj_t.x, sj_t.y),
                            pos_i_after=(si_t.x, si_t.y),
                            pos_j_after=(sj_t.x, sj_t.y),
                        )

                    # For swap, need also t+1
                    if t <= max_len - 1:
                        si_next = self._get_state_at_time(paths[i], t + 1)
                        sj_next = self._get_state_at_time(paths[j], t + 1)

                        if (
                            si_t.x == sj_next.x
                            and si_t.y == sj_next.y
                            and sj_t.x == si_next.x
                            and sj_t.y == si_next.y
                        ):
                            return Conflict(
                                robot_i=i,
                                robot_j=j,
                                time=t,
                                x=si_t.x,
                                y=si_t.y,
                                is_swap=True,
                                pos_i_before=(si_t.x, si_t.y),
                                pos_j_before=(sj_t.x, sj_t.y),
                                pos_i_after=(si_next.x, si_next.y),
                                pos_j_after=(sj_next.x, sj_next.y),
                            )
                        
        return None

    def _collect_constraint_states(
        self, constraints: List[Constraint], robot_id: int
    ) -> Set[State]:
        """
        From the global constraint list, collect only the State objects that
        apply to the given robot. These are fed into state_time_a_star.
        """
        return {c.state for c in constraints if c.robot_id == robot_id}

    def _expand_node(self, node: CTNode, conflict: Conflict) -> List[Optional[CTNode]]:
        """
        For a given conflict, create two child nodes:
        - one with an additional constraint on robot_i
        - one with an additional constraint on robot_j
        """
        children: List[Optional[CTNode]] = []

        # Child 1: constrain robot_i
        c1 = self._constraint_from_conflict(conflict, conflict.robot_i)
        child1 = self._recompute_child(node, c1, conflict.robot_i)
        children.append(child1)

        # Child 2: constrain robot_j
        c2 = self._constraint_from_conflict(conflict, conflict.robot_j)
        child2 = self._recompute_child(node, c2, conflict.robot_j)
        children.append(child2)

        return children

    def _constraint_from_conflict(self, conflict: Conflict, robot_id: int) -> Constraint:
        """
        Turn a conflict into a single Constraint for the given robot.
        For a vertex conflict: forbid the robot from being at (x, y) at time t.
        For a swap conflict: forbid the robot from being at its "after" position
        at time t+1. (This is a simple way to break the swap.)
        """
        if not conflict.is_swap:
            # vertex conflict
            state = State(t=conflict.time, x=conflict.x, y=conflict.y)
        else:
            # swap conflict
            if robot_id == conflict.robot_i:
                x, y = conflict.pos_i_after
            else:
                x, y = conflict.pos_j_after
            state = State(t=conflict.time + 1, x=x, y=y)

        return Constraint(robot_id=robot_id, state=state)

    def _recompute_child(
        self,
        parent: CTNode,
        new_constraint: Constraint,
        robot_id: int,
    ) -> Optional[CTNode]:
        """
        Create a child CT node by adding new_constraint to the parent's
        constraints and re-planning ONLY the given robot's path.
        """
        child_constraints = list(parent.constraints)
        child_constraints.append(new_constraint)

        child_paths = list(parent.paths)

        # Re-plan for the constrained robot
        start_pos = self.starts[robot_id]
        goal_pos = self.goals[robot_id]
        constraint_states = self._collect_constraint_states(child_constraints, robot_id)

        new_path = self.state_time_a_star(
            start_pos=start_pos,
            goal_pos=goal_pos,
            constraints_list=constraint_states,
            env=self.env,
            check_collisions=True,
            max_time=self.max_time,
        )
        if new_path is None:
            # This branch is infeasible
            return None

        child_paths[robot_id] = new_path
        child_cost = Planner.calculate_total_path_cost(child_paths)

        return CTNode(cost=child_cost, constraints=child_constraints, paths=child_paths)

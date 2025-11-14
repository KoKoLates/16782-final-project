
from core import Env
from planner import Planner
from coverage import CoverageOptimizer
from visualizer import Visualizer


if __name__ == "__main__":
    env = Env("example.txt")

    cov_opt = CoverageOptimizer(env)
    targets = cov_opt.process()

    planner = Planner(env)
    paths = planner.process()

    viz = Visualizer(env)
    viz.plot(paths)
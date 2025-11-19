import argparse

from core import Env
from planner import PrioritizedPlanner, CBSPlanner
from coverage import ParticleSwarmOptimizer
from visualizer import Visualizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='filepath to map file')
    args = parser.parse_args()

    env = Env(args.file)
    cov_opt = ParticleSwarmOptimizer(env)

    points = cov_opt.process()

    planner1 = CBSPlanner(env)
    paths = planner1.process(points)

    viz = Visualizer(env)
    viz.animate(paths, interval=400, file_name="./cache/assets/animation.gif")
    
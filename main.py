import argparse

from core import Env
from planner import PrioritizedPlanner, CBSPlanner
from placement import ParticleSwarmOptimizer
from visualizer import Visualizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, help='filepath to map file')
    parser.add_argument('-o', '--output', type=str, 
                        default="./cache/assets/animation.gif",
                        help="File path for GIF output")
    args = parser.parse_args()

    env = Env(args.file)
    cov_opt = ParticleSwarmOptimizer(env)

    points = cov_opt.process()

    planner1 = CBSPlanner(env)
    paths = planner1.process(points)

    viz = Visualizer(env)
    viz.animate(paths, interval=200, file_name=args.output)

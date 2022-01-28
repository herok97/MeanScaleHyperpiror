import os

from solver import Solver
from config import Config
import torch
if __name__ == '__main__':
    config = Config()
    solver = Solver(config)
    solver.build()
    solver.train()
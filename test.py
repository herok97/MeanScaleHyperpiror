import os

from solver import Solver
from config import Config
from pathlib import Path

if __name__ == '__main__':
    config = Config()
    solver = Solver(config, isTrain=False)
    solver.build()
    file = open(config.result_dir / Path('test.txt'), 'w')
    file.write(solver.test())
    file.close()
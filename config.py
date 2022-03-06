import os
from pathlib import Path
class Config():
    def __init__(self):
        # Directory
        self.root_dir = Path(os.getcwd())
        self.save_dir = self.root_dir / Path('save')
        self.result_dir = self.root_dir / Path('result')
        self.log_dir = self. root_dir / Path('log')
        self.dataset = self. root_dir / Path('data')

        # pretrain
        self.pre_train = False
        self.save_model_dir = self.root_dir / Path('save/model.pkl')    # model name

        # training condition
        self.batch_size = 8
        self.epochs = 5000      # infinite
        self.log_step = 250     # logging each 250 steps
        self.total_global_step = 1400000    # total step
        self.lr = 1e-4          # initial learning rate
        self.quality = 7    # quality = 0, 1, 2, ... , 7

        # test/validation
        self.save_step = 50000
        self.val_step = 10000
        self.test_step = 100000
        self.test_batch_size = 1

        # details
        self.num_workers = 4
        self.seed = 1234
        self.clip_max_norm = 5.0

        print(self.__dict__)
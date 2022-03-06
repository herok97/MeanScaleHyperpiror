import random
import time
from tqdm import tqdm
from model import *
from torch.utils.data import DataLoader
from torchvision import transforms
from basics import *
from tensorboardX import SummaryWriter
import numpy as np
import torch
from torch import optim


class Solver():
    def __init__(self, config, isTrain=True):
        # get config
        self.config = config

        # create model
        self.imageCompressor = meanScaleHyperprior(quality=self.config.quality).cuda()  # quality = 0, 1, ... , 7

        # initialization global step
        self.global_step = 0

        # check mode
        self.isTrain = isTrain

        # create tensorboard log writer
        self.writer = SummaryWriter(self.config.log_dir)

        # load datasets
        if self.isTrain:
            print('load train datasets')
            train_transforms = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                ]
            )
            train_dataset = ImageFolder(self.config.dataset, split="train", transform=train_transforms)

            self.train_dataloader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                shuffle=True,
                pin_memory=True
            )

            val_transforms = transforms.Compose(
                [transforms.ToTensor()]
            )

            val_dataset = ImageFolder(self.config.dataset, split="val", transform=val_transforms)

            print('load val datasets')

            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.config.test_batch_size,
                num_workers=self.config.num_workers,
                shuffle=False,
                pin_memory=True,
            )

        test_transforms = transforms.Compose(
            [transforms.ToTensor()]
        )

        test_dataset = ImageFolder(self.config.dataset, split="test", transform=test_transforms)

        print('load test datasets')

        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.config.test_batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def build(self):
        # fix random seed
        fix_seed(self.config.seed)

        # get model parameters and create optimizer
        parameters = self.imageCompressor.parameters()
        self.optimizer = optim.Adam(parameters, lr=self.config.lr)

        # load pre-trained model
        if self.config.pre_train:

            # load checkpoint to continue previous training
            print(f'load pre-trained model {self.config.save_model_dir}')
            if self.isTrain:
                self.load_checkpoint(self.imageCompressor,
                                     [self.optimizer],
                                     self.config.save_model_dir)
            # load model weights only
            else:
                self.load_model(self.imageCompressor, self.config.save_model_dir)

    def train(self):
        self.imageCompressor.train()

        for epoch in range(self.config.epochs):
            tqdm.write(f'Epoch[{epoch}/{self.config.epochs}]')

            for i, img in enumerate(tqdm(self.train_dataloader)):
                self.global_step += 1

                if 1100000 > self.global_step:
                    self.optimizer.param_groups[0]["lr"] = 1e-4
                elif 1100000 <= self.global_step < 1300000:
                    self.optimizer.param_groups[0]["lr"] = 5e-5
                elif 1300000 <= self.global_step < 1350000:
                    self.optimizer.param_groups[0]["lr"] = 1e-5
                elif 1350000 <= self.global_step < 1400000:
                    self.optimizer.param_groups[0]["lr"] = 1e-6

                # until being reached to the total global steps
                if self.config.total_global_step < self.global_step:
                    exit(0)

                x = img.cuda()

                x_hat, y_hat, mse_loss, bpp_feature, bpp_z, bpp = self.imageCompressor(x)

                mse_loss, bpp_feature, bpp_z, bpp = \
                    torch.mean(mse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)

                rd_loss = mse_loss * self.imageCompressor.getLambda() + bpp

                self.optimizer.zero_grad()
                rd_loss.backward()

                clip_gradient(self.optimizer, self.config.clip_max_norm)
                self.optimizer.step()

                psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10))

                bpp, loss, psnr = np.float(bpp.detach()), np.float(rd_loss.detach()), np.float(psnr.detach())

                bpp = np.mean(bpp)
                loss = np.mean(loss)
                psnr = np.mean(psnr)

                if self.global_step % self.config.log_step == 0:
                    # write tensroboard
                    self.writer.add_scalar('bpp', bpp, self.global_step)
                    self.writer.add_scalar('loss', loss, self.global_step)
                    self.writer.add_scalar('psnr', psnr, self.global_step)

                    tqdm.write(
                        f'Step:[{self.global_step}/{self.config.total_global_step}]\t' +
                        f'lr: {self.optimizer.param_groups[0]["lr"]}\n' +
                        f'bpp: {np.round(bpp, 3)}\t' +
                        f'psnr: {np.round(psnr, 3)}\t' +
                        f'loss: {np.round(loss, 3)}\t')

                # validation        (tensorboard)
                if self.global_step % self.config.val_step == 0:
                    self.val()

                # save checkpoint   (saved save folder)
                if self.global_step % self.config.save_step == 0:
                    self.save_checkpoint(epoch, self.imageCompressor, [self.optimizer],
                                         self.global_step,
                                         self.config.save_dir / Path(str(time.time()).split('.')[0] + '_' + str(
                                             self.global_step) + '.pkl'))

                # test  (saved result folder)
                if self.global_step % self.config.test_step == 0:
                    print('----------------------TEST--------------------\n')
                    file = open(self.config.result_dir / Path(f'test{self.global_step}.txt'), 'w')
                    file.write(self.test())
                    file.close()

    def test(self):
        self.imageCompressor.eval()
        with torch.no_grad():
            sumBpp = 0
            sumPsnr = 0
            sumMsssim = 0
            sumMsssimDB = 0
            cnt = 0
            for i, x in enumerate(self.test_dataloader):
                x = x.cuda()
                recon_image, _, _, bpp_feature, bpp_z, bpp = self.imageCompressor(x)
                mse_loss = torch.mean((recon_image - x).pow(2))
                mse_loss, bpp_feature, bpp_z, bpp = \
                    torch.mean(mse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)
                psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10))
                sumBpp += bpp
                sumPsnr += psnr
                msssim = ms_ssim(recon_image.cpu().detach(), x.cpu().detach(), data_range=1.0, size_average=True)
                msssimDB = -10 * (torch.log(1 - msssim) / np.log(10))
                sumMsssimDB += msssimDB
                sumMsssim += msssim
                cnt += 1
            sumBpp /= cnt
            sumPsnr /= cnt
            sumMsssim /= cnt
            sumMsssimDB /= cnt
            print(
                "Dataset Average result---Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(sumBpp,
                                                                                                             sumPsnr,
                                                                                                             sumMsssim,
                                                                                                             sumMsssimDB))
            result = f"{sumBpp:.6f}, {sumPsnr:.6f}, {sumMsssim:.6f}, {sumMsssimDB:.6f}\n"
        self.imageCompressor.train()
        return result

    def val(self):
        print("============validation============")
        self.imageCompressor.eval()
        with torch.no_grad():
            sumBpp = 0
            sumPsnr = 0
            sumMsssim = 0
            sumMsssimDB = 0
            cnt = 0
            for batch_idx, input in enumerate(self.val_dataloader):
                input = input.cuda()
                recon_image, _, _, bpp_feature, bpp_z, bpp = self.imageCompressor(input)
                mse_loss = torch.mean((recon_image - input).pow(2))
                mse_loss, bpp_feature, bpp_z, bpp = \
                    torch.mean(mse_loss), torch.mean(bpp_feature), torch.mean(bpp_z), torch.mean(bpp)
                psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10))
                sumBpp += bpp
                sumPsnr += psnr
                msssim = ms_ssim(recon_image.cpu().detach(), input.cpu().detach(), data_range=1.0, size_average=True)
                msssimDB = -10 * (torch.log(1 - msssim) / np.log(10))
                sumMsssimDB += msssimDB
                sumMsssim += msssim
                cnt += 1

            sumBpp /= cnt
            sumPsnr /= cnt
            sumMsssim /= cnt
            sumMsssimDB /= cnt
            print(
                "Dataset result---Bpp:{:.6f}, PSNR:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(
                    sumBpp, sumPsnr, sumMsssim, sumMsssimDB))
            # write tensroboard
            self.writer.add_scalar('val_bpp', sumBpp, self.global_step)
            self.writer.add_scalar('val_psnr', sumPsnr, self.global_step)
        self.imageCompressor.train()

    def save_checkpoint(self, epoch, model, optimizers, global_step, path):
        os = []
        for optimizer in optimizers:
            os.append(optimizer.state_dict())
        state = {
            'Epoch': epoch,
            'State_dict': model.state_dict(),
            'optimizers': os,
            'Global_step': global_step
        }
        torch.save(state, path)

    def load_checkpoint(self, model, optimizers, path):
        checkpoint = torch.load(path)
        pretrained_dict = checkpoint['State_dict']
        new_model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
                            # and k not in ['priorDecoder.4.weight', 'priorDecoder.4.bias', 'priorEncoder.0.weight', 'Encoder.6.bias', 'Encoder.6.weight', 'Decoder.0.weight']}
                            # for high bpp to low bpp
        print('Load layers\n', pretrained_dict.keys())
        new_model_dict.update(pretrained_dict)
        print('All layers\n', [name for name, _ in model.named_parameters()])
        model.load_state_dict(new_model_dict)

        for i, optimizer in enumerate(optimizers):
            optimizer.load_state_dict(checkpoint['optimizers'][i])
        self.global_step = checkpoint['Global_step']

    def load_model(self, model, path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['State_dict'])


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # adaptivepooling 사용시 RuntimeError
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
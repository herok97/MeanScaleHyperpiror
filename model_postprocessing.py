from basics import *
import math
# import torchac
# import numpy as np
from unet import UNet


class EnhancedMeanScaleHyperprior(nn.Module):
    def __init__(self, N=192, M=320, lmbda=8192):
        super(EnhancedMeanScaleHyperprior, self).__init__()
        self.N = N
        self.M = M
        self.pad_h1 = 0
        self.pad_h2 = 0
        self.pad_w1 = 0
        self.pad_w2 = 0
        self.mxrange = 150
        self.lmbda = lmbda

        self.Encoder = nn.Sequential(
            nn.Conv2d(3, N, 5, stride=2, padding=2),
            GDN(N),
            nn.Conv2d(N, N, 5, stride=2, padding=2),
            GDN(N),
            nn.Conv2d(N, N, 5, stride=2, padding=2),
            GDN(N),
            nn.Conv2d(N, M, 5, stride=2, padding=2),
        )

        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(M, N, 5, stride=2, padding=2, output_padding=1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, 3, 5, stride=2, padding=2, output_padding=1),
        )

        self.priorEncoder = nn.Sequential(
            nn.Conv2d(M, N, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N, N, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(N, N, 5, stride=2, padding=2)
        )

        self.priorDecoder = nn.Sequential(
            nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(N, N * 3 // 2, 5, stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(N * 3 // 2, M * 2, 3, stride=1, padding=1)
        )

        self.entropy_model_z = BitEstimator(N)

        #------------------ postprocessing model ------------------
        self.unet = UNet(n_channels=3, N=M) # called in self.decode()

    def forward(self, img):
        y = self.encode(img)

        if self.training:
            quant_noise_feature = torch.zeros_like(y).cuda()
            quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
            y_hat = y + quant_noise_feature
        else:
            y_hat = torch.round(y)

        z = self.priorEncoder(y)

        if self.training:
            quant_noise_z = torch.zeros_like(z).cuda()
            quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(quant_noise_z), -0.5, 0.5)
            z_hat = z + quant_noise_z
        else:
            z_hat = torch.round(z)

        mean, scale = self.priorDecoder(z_hat).chunk(2, 1)

        x_hat = self.decode(y_hat)

        # distortion
        mse_loss = torch.mean((x_hat - img).pow(2))
        clipped_recon_image = x_hat.clamp(0., 1.)

        def feature_probs_based_sigma(feature, mean, scale):
            scale = torch.exp(torch.clamp(scale, -20, 20))
            gaussian = torch.distributions.laplace.Laplace(mean, scale)
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))

            # if not self.training:
            #     def getrealbitsg(x, gaussian):
            #         # print("NIPS18noc : mn : ", torch.min(x), " - mx : ", torch.max(x), " range : ", self.mxrange)
            #         cdfs = []
            #         x = x + self.mxrange
            #         n, c, h, w = x.shape
            #         for i in range(-self.mxrange, self.mxrange):
            #             i = torch.tensor(i)    # <- 추가
            #             cdfs.append(gaussian.cdf(i - 0.5).view(n, c, h, w, 1))
            #         cdfs = torch.cat(cdfs, 4).cpu().detach()
            #
            #         byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16),
            #                                                check_input_bounds=True)
            #
            #         real_bits = torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda()
            #
            #         sym_out = torchac.decode_float_cdf(cdfs, byte_stream)
            #
            #         return sym_out - self.mxrange, real_bits
            #
            #     decodedx, real_bits = getrealbitsg(feature, gaussian)
            #     total_bits = real_bits

            return total_bits, probs

        def iclr18_estimate_bits_z(z):
            prob = self.entropy_model_z(z + 0.5) - self.entropy_model_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))

            # if not self.training:
            #     def getrealbits(x):
            #         cdfs = []
            #         x = x + self.mxrange
            #         n, c, h, w = x.shape
            #         for i in range(-self.mxrange, self.mxrange):
            #             i = torch.tensor(i)      # <- 추가
            #             cdfs.append(self.entropy_model_z(i - 0.5).view(1, c, 1, 1, 1).repeat(1, 1, h, w, 1))
            #         cdfs = torch.cat(cdfs, 4).cpu().detach()
            #         byte_stream = torchac.encode_float_cdf(cdfs, x.cpu().detach().to(torch.int16),
            #                                                check_input_bounds=True)
            #
            #         real_bits = torch.sum(torch.from_numpy(np.array([len(byte_stream) * 8])).float().cuda())
            #
            #         sym_out = torchac.decode_float_cdf(cdfs, byte_stream)
            #
            #         return sym_out - self.mxrange, real_bits
            #
            #     decodedx, real_bits = getrealbits(z)
            #     total_bits = real_bits

            return total_bits, prob

        total_bits_feature, _ = feature_probs_based_sigma(y_hat, mean, scale)
        total_bits_z, _ = iclr18_estimate_bits_z(z_hat)

        im_shape = img.size()
        bpp_feature = total_bits_feature / (im_shape[0] * im_shape[2] * im_shape[3])
        bpp_z = total_bits_z / (im_shape[0] * im_shape[2] * im_shape[3])
        bpp = bpp_feature + bpp_z
        return clipped_recon_image, y_hat, mse_loss, bpp_feature, bpp_z, bpp

    def encode(self, img):
        img_shape = img.size()
        h, w = img_shape[2], img_shape[3]

        if h % 64 != 0:
            pad_length = 64 - h % 64
            self.pad_h1 = pad_length // 2
            self.pad_h2 = pad_length - pad_length // 2
            pad_h = nn.ReflectionPad2d((0, 0, self.pad_h1, self.pad_h2,))
            img = pad_h(img)

        if w % 64 != 0:
            pad_length = 64 - w % 64
            self.pad_w1 = pad_length // 2
            self.pad_w2 = pad_length - pad_length // 2
            pad_w = nn.ReflectionPad2d((self.pad_w1, self.pad_w2, 0, 0))
            img = pad_w(img)

        return self.Encoder(img)

    def decode(self, y_hat):
        img = self.Decoder(y_hat)
        img = self.unet(img)
        if self.pad_h1 + self.pad_h2 == 0:
            if self.pad_w1 + self.pad_w2 == 0:
                pass
            else:
                img = img[:, :, :, self.pad_w1:-self.pad_w2]
        else:
            if self.pad_w1 + self.pad_w2 == 0:
                img = img[:, :, self.pad_h1:-self.pad_h2, :]
            else:
                img = img[:, :, self.pad_h1:-self.pad_h2, self.pad_w1:-self.pad_w2]

        # initialization padding params
        self.pad_h1 = 0
        self.pad_h2 = 0
        self.pad_w1 = 0
        self.pad_w2 = 0
        return img

    def getLambda(self):
        return self.lmbda

def enhanbedMeanScaleHyperprior(quality):
    lmbdas = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    if quality < 4:
        N = 192
        M = 192
    else:
        N = 192
        M = 320

    return EnhancedMeanScaleHyperprior(N, M, lmbdas[quality])


if __name__ == '__main__':
    pass

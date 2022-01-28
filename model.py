from basics import *
import math


# import torchac

class MeanScaleHyperprior(nn.Module):
    def __init__(self, N=192, M=320, lmbda=8192):
        super(MeanScaleHyperprior, self).__init__()
        self.N = N
        self.M = M
        self.pad_h = 0
        self.pad_w = 0
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
            return total_bits, probs

        def iclr18_estimate_bits_z(z):
            prob = self.entropy_model_z(z + 0.5) - self.entropy_model_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
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
            self.pad_h = (64 - h % 64) // 2
            pad_h = nn.ReflectionPad2d((0, 0, self.pad_h, self.pad_h,))
            img = pad_h(img)

        if w % 64 != 0:
            self.pad_w = (64 - w % 64) // 2
            pad_w = nn.ReflectionPad2d((self.pad_w, self.pad_w, 0, 0))
            img = pad_w(img)

        return self.Encoder(img)

    def decode(self, y_hat):
        img = self.Decoder(y_hat)
        if self.pad_h == 0:
            if self.pad_w == 0:
                pass
            else:
                img = img[:, :, :, self.pad_w:-self.pad_w]
        else:
            if self.pad_w == 0:
                img = img[:, :, self.pad_h:-self.pad_h, :]

            else:
                img = img[:, :, self.pad_h:-self.pad_h, self.pad_w:-self.pad_w]

        # initialization padding params
        self.pad_h = 0
        self.pad_w = 0
        return img

    def getLambda(self):
        return self.lmbda


def meanScaleHyperprior(quality):
    lmbdas = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    if quality < 4:
        N = 192
        M = 192
    else:
        N = 192
        M = 320

    return MeanScaleHyperprior(N, M, lmbdas[quality])


if __name__ == '__main__':
    pass

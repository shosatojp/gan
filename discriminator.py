import torch.nn as nn
import torch


class Discriminator(nn.Module):
    def __init__(self, image_size):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=image_size,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=image_size,
                      out_channels=image_size*2,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=image_size*2,
                      out_channels=image_size*4,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=image_size*4,
                      out_channels=image_size*8,
                      kernel_size=4,
                      stride=2,
                      padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.last = nn.Conv2d(in_channels=image_size*8,
                              out_channels=1,
                              kernel_size=4,
                              stride=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.last(out)
        return out


if __name__ == '__main__':
    D = Discriminator(image_size=64)
    z = torch.randn(1, 20).view(1, 20, 1, 1)
    from generator import Generator
    G = Generator()
    fake = G(z)
    out = D(fake)
    print(nn.Sigmoid()(out))

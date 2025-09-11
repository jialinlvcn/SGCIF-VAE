import torch
import torch.nn as nn
from model.standard import opt_backbone


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0),
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)


class CDVAE(nn.Module):
    def __init__(
        self, d, z, with_classifier=True, mtype="resnet", task="SOC", **kwargs
    ):
        super(CDVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, d // 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 8, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 8, d // 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 4, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 4, d // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d // 2, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d, track_running_stats=False),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d, track_running_stats=False),
            ResBlock(d, d, bn=True),
        )

        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d, track_running_stats=False),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d, track_running_stats=False),
            nn.ConvTranspose2d(
                d, d // 2, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(d // 2, track_running_stats=False),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(
                d // 2, d // 4, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(d // 4, track_running_stats=False),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(
                d // 4, d // 8, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(d // 8, track_running_stats=False),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(
                d // 8, 1, kernel_size=4, stride=2, padding=1, bias=False
            ),
        )
        self.xi_bn = nn.BatchNorm2d(1, track_running_stats=False)

        self.f = 8
        self.d = d
        self.z = z
        self.fc11 = nn.Linear(d * self.f**2, self.z)
        self.fc12 = nn.Linear(d * self.f**2, self.z)
        self.fc21 = nn.Linear(self.z, d * self.f**2)
        self.conv21 = nn.Conv2d(1, 1, kernel_size=1, stride=1, bias=False)
        self.conv22 = nn.Conv2d(1, 1, kernel_size=1, stride=1, bias=False)
        self.with_classifier = with_classifier
        if self.with_classifier:
            self.classifier = opt_backbone(mtype, task)

    def encode(self, x):
        h = self.encoder(x)
        h1 = h.view(x.shape[0], self.d * self.f**2)
        return h.flatten(start_dim=1), self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x):
        _, mu, logvar = self.encode(x)
        hi = self.reparameterize(mu, logvar)
        hi_projected = self.fc21(hi)
        xi = self.decode(hi_projected)
        xi = self.xi_bn(xi)

        if self.with_classifier:
            out = self.classifier(x - xi)
            return out, hi_projected, xi, mu, logvar
        else:
            return xi

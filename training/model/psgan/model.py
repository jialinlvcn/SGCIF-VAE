import torch
import torch.nn as nn
import torch.nn.functional as F
from model.standard import opt_backbone
import functools


class PSDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, cla, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(PSDiscriminator, self).__init__()
        if (
            type(norm_layer) == functools.partial  # noqa: E721
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
            if n == 1:
                model1 = sequence + [nn.MaxPool2d(4, 4)]
            elif n == 2:
                model2 = sequence + [nn.MaxPool2d(2, 2)]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=2,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        model3 = sequence
        self.feat1 = nn.Sequential(*model1)
        self.feat2 = nn.Sequential(*model2)
        self.feat3 = nn.Sequential(*model3)

        # self.fc = nn.Linear(8**n_layers * ndf * nf_mult_prev, 1)  # output 1 channel prediction map
        self.fc = nn.Linear(32768, 1)
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        device = self.model[0].weight.device
        input = input.to(device)
        x = self.model(input)
        return self.fc(x.view(x.size(0), -1)), torch.cat(
            [self.feat1(input) / 8.0, self.feat2(input) / 4.0, self.feat3(input)], 1
        )


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=["avg", "max"]):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                avg_pool = F.avg_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == "max":
                max_pool = F.max_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == "lp":
                lp_pool = F.lp_pool2d(
                    x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == "lse":
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = (
            torch.nn.Sigmoid()(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        )
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.nn.Sigmoid()(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
        pool_types=["avg", "max"],
        no_spatial=False,
    ):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class PSGenerator(nn.Module):
    """perturbation-seeking generator proposed by

    G. Cheng, X. Sun, K. Li, L. Guo, and J. Han, "Perturbation Seeking Generative
    Adversarial Networks: A Defense Framework for Remote Sensing Image Scene Classification", IEEE Trans. Geosci. Remote Sens.

    """

    def __init__(self, input_nc, output_nc, cla, ngf=64, norm_layer=nn.BatchNorm2d):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        super(PSGenerator, self).__init__()
        if type(norm_layer) == functools.partial:  # noqa: E721
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc + cla, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]
        model += [CBAM(ngf, 16)]
        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2**i
            model += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=use_bias,
                ),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True),
            ]

        down = model
        mult = 2**n_downsampling
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if i == 0:
                up = [
                    nn.ConvTranspose2d(
                        ngf * mult + cla,
                        int(ngf * mult / 2),
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=use_bias,
                    ),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True),
                ]
                up += [CBAM(int(ngf * mult / 2), 16)]
            else:
                up += [
                    nn.ConvTranspose2d(
                        ngf * mult,
                        int(ngf * mult / 2),
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=use_bias,
                    ),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True),
                ]
        up += [nn.ReflectionPad2d(3)]
        up += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        up += [nn.Tanh()]

        self.down = nn.Sequential(*down)
        self.up = nn.Sequential(*up)

    def forward(self, input, label):
        """Standard forward"""
        l = label.view(label.size(0), label.size(1), 1, 1).expand(  # noqa: E741
            label.size(0), label.size(1), input.size(2), input.size(3)
        )
        cinput = torch.cat([input, l], 1)
        down = self.down(cinput)
        l = label.view(label.size(0), label.size(1), 1, 1).expand(  # noqa: E741
            label.size(0), label.size(1), down.size(2), down.size(3)
        )
        cdown = torch.cat([down, l], 1)
        return self.up(cdown)


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type="instance"):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "none":
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == "lsgan":
            self.loss = nn.MSELoss()
        elif gan_mode == "vanilla":
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ["wgangp"]:
            self.loss = None
        else:
            raise NotImplementedError("gan mode %s not implemented" % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ["lsgan", "vanilla"]:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == "wgangp":
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class PSGANModel(nn.Module):
    def __init__(
        self,
        ndf,
        ngf,
        lr,
        mtype="resnet",
        task="SOC",
        cla=None,
        norm="batch",
        input_nc=1,
        output_nc=1,
        with_classifier=True,
        n_layers_D=3,
        is_train=True,
        device="cpu",
        Dfeat_lambda=100,
        l1_lambda=50,
        **kwargs,
    ) -> None:
        super(PSGANModel, self).__init__()
        self.is_train = is_train
        self.alpha = 0.9
        self.with_classifier = with_classifier
        if self.with_classifier:
            self.backbone = opt_backbone(mtype, task)

        if mtype == "resnet":
            if task == "MSTAR_SOC":
                self.cla = 10
            elif task == "SAR_ACD":
                self.cla = 6
            else:
                self.cla = 4
        elif mtype == "vgg":
            if task == "MSTAR_SOC":
                self.cla = 10
            elif task == "SAR_ACD":
                self.cla = 6
            else:
                self.cla = 4

        self.netD = None
        norm_layerD = get_norm_layer(norm_type=norm)
        self.netD = PSDiscriminator(
            input_nc + output_nc,
            self.cla,
            ndf,
            n_layers=n_layers_D,
            norm_layer=norm_layerD,
        )

        self.netG = None
        norm_layerG = get_norm_layer(norm_type=norm)
        self.netG = PSGenerator(
            input_nc, output_nc, self.cla, ngf, norm_layer=norm_layerG
        )
        self.l1_lambda = l1_lambda
        self.Dfeat_lambda = Dfeat_lambda
        self.device = device
        self.criterionGAN = GANLoss("lsgan").to(self.device)
        self.Dcla = nn.CrossEntropyLoss()
        self.criterionL1 = torch.nn.L1Loss()

        self.optimizer_G = torch.optim.Adam(
            self.netG.parameters(), lr=lr, betas=(0.5, 0.9)
        )
        self.optimizer_D = torch.optim.Adam(
            self.netD.parameters(), lr=lr * 4, betas=(0.5, 0.9)
        )
        self.optimizer_C = torch.optim.SGD(
            self.backbone.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
        )

    def set_input(self, adv_x, x, y):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real_A = adv_x
        self.real_B = x
        self.label = y

    def forward(self, x=None):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        if not self.is_train:
            # self.backbone.eval()
            # self.one_hot_true = torch.zeros([self.label.shape[0], self.cla]).to(
            #     self.device
            # )
            # self.noise = self.alpha * self.netG(x, self.one_hot_true)
            # self.fake_B = torch.clamp((x - self.noise), -1, 1)
            # adv_logits = self.backbone(self.fake_B)
            # return adv_logits
            try:
                return self.backbone(x)
            except:  # noqa: E722
                return self.backbone(self.real_A)
        else:
            self.backbone.train()
            self.one_hot_true = torch.zeros([self.label.shape[0], self.cla]).to(
                self.device
            )
            for i in range(self.real_B.shape[0]):
                self.one_hot_true[i, self.label[i]] = 1
            self.noise = self.alpha * self.netG(self.real_A, self.one_hot_true)  # G(A)
            self.fake_B = torch.clamp((self.real_A - self.noise), -1, 1)
            self.r_noise = torch.clamp(
                (self.real_A - self.real_B.to(self.device)), -1, 1
            )

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        adv_AB = torch.cat(
            (self.real_A, self.real_B.to(self.device)), 1
        )  # we use conditional GANs; we need to feed both input and output to the discriminator
        adv_fake, _ = self.netD(adv_AB)
        self.loss_D_adv = self.criterionGAN(adv_fake, False)

        real_AB = torch.cat((self.real_B, self.real_B), 1)
        pred_real, self.real_feat = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        fake_AB = torch.cat((self.fake_B.detach(), self.real_B.to(self.device)), 1)
        pred_fake, _ = self.netD(fake_AB)
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_adv + self.loss_D_real + self.loss_D_fake) * 1
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.fake_B, self.real_B.to(self.device)), 1)
        pred_fake, fake_feat = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, || noise - r_noise ||_1
        self.loss_G_L1 = self.criterionL1(self.noise, self.r_noise)

        # Third, ||real_Dfeat - fake_Dfeat ||_1
        self.loss_G_Dfeat = nn.L1Loss()(self.real_feat, fake_feat)

        # combine loss and calculate gradients
        self.loss_G = (
            1 * self.loss_G_GAN
            + self.l1_lambda * self.loss_G_L1
            + self.Dfeat_lambda * self.loss_G_Dfeat
        )

    def backward_C(self):
        adv_logits = self.backbone(self.real_A)
        self.loss_C_adv = F.cross_entropy(adv_logits, self.label.to(self.device))

        noise = self.alpha * self.netG(self.real_A, self.one_hot_true)
        fake_B = torch.clamp((self.real_A - noise), -1, 1)
        fake_logits = self.backbone(fake_B)
        self.loss_C_fake = F.cross_entropy(fake_logits, self.label.to(self.device))

        real_logits = self.backbone(self.real_B.to(self.device))
        self.loss_C_real = F.cross_entropy(real_logits, self.label.to(self.device))

        # combine loss and calculate gradients
        self.loss_C = self.loss_C_adv + self.loss_C_fake + self.loss_C_real
        self.loss_C.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(
            self.netD, False
        )  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights
        # update C
        self.set_requires_grad(
            self.netD, False
        )  # D requires no gradients when optimizing C
        self.optimizer_C.zero_grad()  # set C's gradients to zero
        self.backward_C()  # calculate graidents for C
        self.optimizer_C.step()  # udpate C's weights

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

import torch


def activation_func(activation_fn):

    if 'linear' in activation_fn:
        return None
    elif 'leaky_relu' in activation_fn:
        return torch.nn.LeakyReLU(negative_slope=0.10, inplace=True)
    elif 'relu' in activation_fn:
        return torch.nn.ReLU()
    elif 'elu' in activation_fn:
        return torch.nn.ELU()
    elif 'sigmoid' in activation_fn:
        return torch.nn.Sigmoid()
    else:
        raise ValueError('Unsupported activation function: {}'.format(activation_fn))


'''
Network layers
'''
class Conv2d(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(Conv2d, self).__init__()

        padding = kernel_size // 2

        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)

        # Select the type of weight initialization, by default kaiming_uniform
        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.conv.weight)
        elif weight_initializer == 'kaiming_uniform':
            pass
        else:
            raise ValueError('Unsupported weight initializer: {}'.format(weight_initializer))

        self.activation_func = activation_func

        assert not (use_batch_norm and use_instance_norm), \
            'Unable to apply both batch and instance normalization'

        self.use_norm = use_batch_norm or use_instance_norm

        if use_batch_norm:
            self.norm = torch.nn.BatchNorm2d(out_channels)
        elif use_instance_norm:
            self.norm = torch.nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        '''
        Forward input x through a convolution layer

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        conv = self.conv(x)
        conv = self.norm(conv) if self.use_norm else conv

        if self.activation_func is not None:
            return self.activation_func(conv)
        else:
            return conv


class TransposeConv2d(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(TransposeConv2d, self).__init__()

        padding = kernel_size // 2

        self.deconv = torch.nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            output_padding=1,
            bias=False)

        # Select the type of weight initialization, by default kaiming_uniform
        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.deconv.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.deconv.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.deconv.weight)
        elif weight_initializer == 'kaiming_uniform':
            pass
        else:
            raise ValueError('Unsupported weight initializer: {}'.format(weight_initializer))

        self.activation_func = activation_func

        assert not (use_batch_norm and use_instance_norm), \
            'Unable to apply both batch and instance normalization'

        self.use_norm = use_batch_norm or use_instance_norm

        if use_batch_norm:
            self.norm = torch.nn.BatchNorm2d(out_channels)
        elif use_instance_norm:
            self.norm = torch.nn.InstanceNorm2d(out_channels)

    def forward(self, x):

        deconv = self.deconv(x)
        deconv = self.norm(deconv) if self.use_norm else deconv

        if self.activation_func is not None:
            return self.activation_func(deconv)
        else:
            return deconv


class UpConv2d(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(UpConv2d, self).__init__()

        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

    def forward(self, x, shape):

        upsample = torch.nn.functional.interpolate(x, size=shape, mode='nearest')
        conv = self.conv(upsample)
        return conv


'''
Network encoder blocks
'''
class ResNetBlock(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(ResNetBlock, self).__init__()

        self.activation_func = activation_func

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=None,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        self.conv3 = None


        if stride != 1 or in_channels != out_channels:
            self.projection = Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=1,
                                     stride=stride,
                                    weight_initializer=weight_initializer,
                                    activation_func=None,
                                    use_batch_norm=use_batch_norm,
                                    use_instance_norm=use_instance_norm)
        else:
            self.projection = None


    def forward(self, x):

        funct = self.conv1(x)
        funct = self.conv2(funct)


        if self.projection and funct.shape != x.shape:
            x = self.projection(x)
        return self.activation_func(funct + x)

        # return None

class VGGNetBlock(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 n_convolution=1,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False,
                 use_maxpool=True):

        super(VGGNetBlock, self).__init__()

        layers = []

        layers.append(Conv2d(in_channels,
                             out_channels,
                             stride=stride,
                             weight_initializer=weight_initializer,
                             activation_func=activation_func,
                             use_batch_norm=use_batch_norm,
                             use_instance_norm=use_instance_norm))

        for i in range(n_convolution - 1):
            layers.append(Conv2d(out_channels,
                                 out_channels,
                                 stride=stride,
                                 weight_initializer=weight_initializer,
                                activation_func=activation_func,
                                use_batch_norm=use_batch_norm,
                                use_instance_norm=use_instance_norm))


        if (use_maxpool == True):
            layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_block = torch.nn.Sequential(*layers)

    def forward(self, x):

        return self.conv_block(x)

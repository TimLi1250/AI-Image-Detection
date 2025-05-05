import torch
import net_utils

class ResNet18Encoder(torch.nn.Module):

    def __init__(self,
                 input_channels=3,
                 n_filters=[32, 64, 128, 256, 256],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(ResNet18Encoder, self).__init__()

        assert len(n_filters) == 5

        activation_func = net_utils.activation_func(activation_func)

        self.ResNetBlocks = torch.nn.ModuleList()


        self.ResNetBlocks.append(net_utils.Conv2d(input_channels, n_filters[0], kernel_size = 7, stride = 2, weight_initializer=weight_initializer,
                                                activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm))
        self.ResNetBlocks.append(torch.nn.MaxPool2d(kernel_size=3, stride=2))


        self.ResNetBlocks.append(net_utils.ResNetBlock(n_filters[0], n_filters[0], weight_initializer=weight_initializer,
                                                activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm))
        self.ResNetBlocks.append(net_utils.ResNetBlock(n_filters[0], n_filters[1], weight_initializer=weight_initializer,
                                                activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm))

        self.ResNetBlocks.append(net_utils.ResNetBlock(n_filters[1], n_filters[1], stride = 2, weight_initializer=weight_initializer,
                                                activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm))
        self.ResNetBlocks.append(net_utils.ResNetBlock(n_filters[1], n_filters[2], stride = 2, weight_initializer=weight_initializer,
                                                activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm))

        self.ResNetBlocks.append(net_utils.ResNetBlock(n_filters[2], n_filters[2], stride = 2, weight_initializer=weight_initializer,
                                                activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm))
        self.ResNetBlocks.append(net_utils.ResNetBlock(n_filters[2], n_filters[3], stride = 2, weight_initializer=weight_initializer,
                                                activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm))

        self.ResNetBlocks.append(net_utils.ResNetBlock(n_filters[3], n_filters[3], stride = 2, weight_initializer=weight_initializer,
                                                activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm))
        self.ResNetBlocks.append(net_utils.ResNetBlock(n_filters[3], n_filters[4], stride = 2, weight_initializer=weight_initializer,
                                                activation_func=activation_func, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm))
    def forward(self, x):

        layers = [x]

        # TODO: Implement forward function
        numblocks = len(self.ResNetBlocks)

        for i in range(numblocks):
            x = self.ResNetBlocks[i](x)

            layers.append(x)

        # Return latent and intermediate features
        return layers[-1], layers[1:-1]

class VGGNet11Encoder(torch.nn.Module):

    def __init__(self,
                 input_channels=3,
                 n_filters=[64, 128, 256, 512, 512], #changed
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(VGGNet11Encoder, self).__init__()

        activation_func = net_utils.activation_func(activation_func)

        # changed last one to 1 because dimensions were collapsing to 0x0
        self.convolutions = [1, 1, 2, 2, 2]
        self.VGGNetBlocks = torch.nn.ModuleList()

        for i in range(5):
          if i == 0:

            self.VGGNetBlocks.append(net_utils.VGGNetBlock(input_channels, n_filters[i], self.convolutions[i],
                                                               weight_initializer=weight_initializer,
                                            use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm))
          elif i == 4 or i == 5:
              self.VGGNetBlocks.append(net_utils.VGGNetBlock(n_filters[i-1], n_filters[i], self.convolutions[i],
                                       weight_initializer=weight_initializer, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm, use_maxpool=False))
          else:
              self.VGGNetBlocks.append(net_utils.VGGNetBlock(n_filters[i-1], n_filters[i], self.convolutions[i],
                                            weight_initializer=weight_initializer, use_batch_norm=use_batch_norm, use_instance_norm=use_instance_norm))


    def forward(self, x):

        layers = [x]

        numblocks = len(self.VGGNetBlocks)
        for i in range(numblocks):
          x = self.VGGNetBlocks[i](x)
          layers.append(x)

        return layers[-1], layers[1:-1]

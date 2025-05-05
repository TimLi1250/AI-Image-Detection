import torch
import networks
from matplotlib import pyplot as plt

class ClassificationModel(object):
    def __init__(self,
                 encoder_type,
                 device=torch.device('cuda')):

        self.device = device

        if encoder_type == 'vggnet11':
            self.encoder = networks.VGGNet11Encoder(n_filters=[64, 128, 256, 512, 512])
            self.decoder = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d((1,1)),
                torch.nn.Flatten(start_dim=1),
                torch.nn.Linear(512, 4096),
                torch.nn.Linear(4096, 4096),
                torch.nn.Linear(4096, 10)
            )
        elif encoder_type == 'resnet18':
            self.encoder = networks.ResNet18Encoder(n_filters=[64, 64, 128, 256, 512])
            self.decoder = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten(start_dim=1),
                torch.nn.Linear(512, 10))
        else:
            raise ValueError('Unsupported encoder type: {}'.format(encoder_type))

        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

    def transform_input(self, images):
        if self.encoder_type == 'vggnet11':
            pass
        elif self.encoder_type == 'resnet18':
            pass

        return None

    def forward(self, image):

        latent, _ = self.encoder(image)
        out = self.decoder(latent)
        return out

    def compute_loss(self, output, label):

        loss_function = torch.nn.CrossEntropyLoss()
        loss = loss_function(output, label)

        loss_info = {
            'loss' : loss
        }
        return loss, loss_info

    def parameters(self):

        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def train(self):

        self.encoder.train()
        self.decoder.train()

    def eval(self):

        self.encoder.eval()
        self.decoder.eval()

    def to(self, device):

        self.device = device
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

    def data_parallel(self):

        self.encoder = torch.nn.DataParallel(self.encoder)
        self.decoder = torch.nn.DataParallel(self.decoder)

    def restore_model(self, restore_path, optimizer=None):

        checkpoint = torch.load(restore_path)

        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(checkpoint['decoder_state_dict'])

        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            return checkpoint['step'], optimizer

    def save_model(self, checkpoint_path, step, optimizer=None):

        checkpoint = {}


        checkpoint['encoder_state_dict'] = self.encoder.state_dict()
        checkpoint['decoder_state_dict'] = self.decoder.state_dict()

        checkpoint['step'] = step

        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        torch.save(checkpoint, checkpoint_path)

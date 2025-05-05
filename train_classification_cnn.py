
import os, argparse
import torch, torchvision
from torchvision import transforms as T
from classification_model import ClassificationModel
from classification_cnn import train

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

# Hyperparameters
parser.add_argument('--n_batch', type=int, required=True)
parser.add_argument('--dataset', type=str, required=True, help="custom")
parser.add_argument('--data_dir', type=str, default=None, help="Root folder with train/ (for custom dataset)")
parser.add_argument('--encoder_type', type=str, required=True, help='vggnet11 | resnet18')
parser.add_argument('--n_epoch', type=int, required=True)
parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--learning_rate_decay', type=float, required=True)
parser.add_argument('--learning_rate_period', type=int, required=True)
parser.add_argument('--checkpoint_path', type=str, required=True)
parser.add_argument('--device', type=str, default='cuda')

args = parser.parse_args()

def build_dataloader():

    if args.dataset == 'custom':
        assert args.data_dir is not None, "--data_dir must be specified for custom dataset"
        train_root = os.path.join(args.data_dir, 'train')
        tf_train = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor()
        ])
        dset = ImageFolder(root=train_root, transform=tf_train)
    else:
        raise ValueError('Unsupported dataset option')

    loader = DataLoader(
        dset, batch_size=args.n_batch, shuffle=True, drop_last=True, num_workers=2)

    return loader, dset.classes


if __name__ == '__main__':

    dataloader_train, class_names = build_dataloader()
    n_class = len(class_names)


    input_channels = 3
    n_input_feature = 3 * 128 * 128
    # Build model and adjust the FC layer to match n_class
    model = ClassificationModel(args.encoder_type, args.device)

    # Replace last Linear if output dim != n_class
    if hasattr(model, 'decoder') and isinstance(model.decoder[-1], torch.nn.Linear):
        if model.decoder[-1].out_features != n_class:
            in_feat = model.decoder[-1].in_features
            model.decoder[-1] = torch.nn.Linear(in_feat, n_class).to(model.device)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

    model.train()
    train(
        model,
        dataloader_train,
        args.n_epoch,
        optimizer,
        args.learning_rate_decay,
        args.learning_rate_period,
        args.checkpoint_path,
        args.device
    )

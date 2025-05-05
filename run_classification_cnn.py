
import os
import argparse
import torch
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from classification_model import ClassificationModel
from classification_cnn import evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True,
                    help='Directory that contains train/, val/ or test/ splits')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='Path to a saved .pth checkpoint file')
parser.add_argument('--encoder_type', type=str, required=True,
                    help='Encoder backbone: resnet18, vggnet11, etc.')
parser.add_argument('--n_batch', type=int, default=64,
                    help='Batch size for evaluation')
parser.add_argument('--device', type=str, default='cuda',
                    help='cuda | cpu')
parser.add_argument('--output_path', type=str, default='./eval_output',
                    help='Folder to dump any evaluation visualisations')
args = parser.parse_args()

os.makedirs(args.output_path, exist_ok=True)

if __name__ == '__main__':

    transforms_eval = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
    ])

    test_root = os.path.join(args.data_dir, 'test')
    if not os.path.isdir(test_root):
        raise FileNotFoundError(f"Could not find a 'test' folder under {args.data_dir}")

    dataset_test = ImageFolder(root=test_root, transform=transforms_eval)
    dataloader_test = DataLoader(dataset_test,
                                 batch_size=args.n_batch,
                                 shuffle=False,
                                 drop_last=False,
                                 num_workers=2)

    class_names = dataset_test.classes
    n_class = len(class_names)

    model = ClassificationModel(args.encoder_type, args.device)

    # If the existing decoder head has > n_class outputs, replace it
    if model.decoder[-1].out_features != n_class:
        in_features = model.decoder[-1].in_features
        model.decoder[-1] = torch.nn.Linear(in_features, n_class)


    checkpoint = model.restore_model(args.checkpoint)

    model.eval()
    evaluate(model, dataloader_test, class_names, args.output_path, args.device)

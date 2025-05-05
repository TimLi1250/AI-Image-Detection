import torch, torchvision, os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

def train(model,
          dataloader,
          n_epoch,
          optimizer,
          learning_rate_decay,
          learning_rate_decay_period,
          checkpoint_path,
          device):

    device = 'cuda' if device == 'gpu' or device == 'cuda' else 'cpu'
    device = torch.device(device)

    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Set up checkpoint and event paths
    model_checkpoint_path = os.path.join(checkpoint_path, 'model-{}.pth')
    event_path = os.path.join(checkpoint_path, 'events')

    train_summary_writer = SummaryWriter(event_path + '-train')

    model.to(device)

    for epoch in range(n_epoch):

        # Accumulate total loss for each epoch
        total_loss = 0.0

        if epoch and epoch % learning_rate_decay_period == 0:

            for param_group in optimizer.param_groups:
                param_group['lr'] *= learning_rate_decay

        for batch, (images, labels) in enumerate(dataloader):

            images = images.to(device)
            labels = labels.to(device)


            outputs = model.forward(images)

            optimizer.zero_grad()

            loss, loss_info = model.compute_loss(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        mean_loss = total_loss / float(batch)

        # Log average loss over the epoch
        print('Epoch={}/{}  Loss: {:.3f}'.format(epoch + 1, n_epoch, mean_loss))

        model.save_model(model_checkpoint_path.format(epoch), epoch, optimizer)

    return model

def evaluate(model, dataloader, class_names, output_path, device):

    device = 'cuda' if device == 'gpu' or device == 'cuda' else 'cpu'
    device = torch.device(device)

    model.to(device)

    n_correct = 0
    n_sample = 0

    with torch.no_grad():

        for (images, labels) in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model.forward(images)

            outputs = torch.argmax(outputs, dim=1)

            # Accumulate number of samples
            n_sample += len(outputs)

            n_correct += torch.sum(outputs == labels).item()

    # TODO: Compute mean accuracy
    mean_accuracy = (n_correct / n_sample) * 100

    print('Mean accuracy over {} images: {:.3f}%'.format(n_sample, mean_accuracy))
import os
import torch
from tensorboardX import SummaryWriter

use_cuda = torch.cuda.is_available()

default_checkpoint = {
    "epoch": 0,
    "train_losses": [],
    "train_accuracy": [],
    "validation_losses": [],
    "validation_accuracy": [],
    "lr": [],
    "grad_norm": [],
    "model": {},
}


def save_checkpoint(checkpoint, dir="./checkpoints", prefix=""):
    # Padded to 4 digits because of lexical sorting of numbers.
    # e.g. 0009.pth
    filename = "{num:0>4}.pth".format(num=checkpoint["epoch"])
    if not os.path.exists(os.path.join(prefix, dir)):
        os.makedirs(os.path.join(prefix, dir))
    torch.save(checkpoint, os.path.join(prefix, dir, filename))


def load_checkpoint(path, cuda=use_cuda):
    if cuda:
        return torch.load(path)
    else:
        # Load GPU model on CPU
        return torch.load(path, map_location=lambda storage, loc: storage)


def init_tensorboard(name="", base_dir="./tensorboard"):
    return SummaryWriter(os.path.join(name, base_dir))


def write_tensorboard(
    writer,
    epoch,
    grad_norm,
    train_loss,
    train_accuracy,
    validation_loss,
    validation_accuracy,
    model,
):
    writer.add_scalar("train_loss", train_loss, epoch)
    writer.add_scalar("train_accuracy", train_accuracy, epoch)
    writer.add_scalar("validation_loss", validation_loss, epoch)
    writer.add_scalar("validation_accuracy", validation_accuracy, epoch)
    writer.add_scalar("grad_norm", grad_norm, epoch)

    for name, param in model.encoder.named_parameters():
        writer.add_histogram(
            "encoder/{}".format(name), param.detach().cpu().numpy(), epoch
        )
        if param.grad is not None:
            writer.add_histogram(
                "encoder/{}/grad".format(name), param.grad.detach().cpu().numpy(), epoch
            )

    for name, param in model.decoder.named_parameters():
        writer.add_histogram(
            "decoder/{}".format(name), param.detach().cpu().numpy(), epoch
        )
        if param.grad is not None:
            writer.add_histogram(
                "decoder/{}/grad".format(name), param.grad.detach().cpu().numpy(), epoch
            )

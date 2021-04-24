import os
import argparse
import multiprocessing
import numpy as np
import random
import time
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from checkpoint import (
    default_checkpoint,
    load_checkpoint,
    save_checkpoint,
    init_tensorboard,
    write_tensorboard,
)
from psutil import virtual_memory

from flags import Flags
from utils import get_network, get_optimizer
from dataset import dataset_loader, START, PAD
from scheduler import CircularLRBeta


def token_to_string(tokens, data_loader):
    result = []
    for example in tokens:
        string = ""
        for token in example:
            token = token.item()
            if token != -1:
                string += data_loader.dataset.id_to_token[token] + " "
        result.append(string)
    return result


def run_epoch(
    data_loader,
    model,
    epoch_text,
    criterion,
    optimizer,
    lr_scheduler,
    teacher_forcing_ratio,
    max_grad_norm,
    device,
    train=True,
):
    # Disables autograd during validation mode
    torch.set_grad_enabled(train)
    if train:
        model.train()
    else:
        model.eval()

    losses = []
    grad_norms = []
    correct_symbols = 0
    total_symbols = 0

    with tqdm(
        desc="{} ({})".format(epoch_text, "Train" if train else "Validation"),
        total=len(data_loader.dataset),
        dynamic_ncols=True,
        leave=False,
    ) as pbar:
        for d in data_loader:
            input = d["image"].to(device)

            # The last batch may not be a full batch
            curr_batch_size = len(input)
            expected = d["truth"]["encoded"].to(device)

            # Replace -1 with the PAD token
            expected[expected == -1] = data_loader.dataset.token_to_id[PAD]

            output = model(input, expected, train, teacher_forcing_ratio)

            decoded_values = output.transpose(1, 2)
            _, sequence = torch.topk(decoded_values, 1, dim=1)
            sequence = sequence.squeeze(1)

            loss = criterion(decoded_values, expected[:, 1:])

            if train:
                optim_params = [
                    p
                    for param_group in optimizer.param_groups
                    for p in param_group["params"]
                ]
                optimizer.zero_grad()
                loss.backward()
                # Clip gradients, it returns the total norm of all parameters
                grad_norm = nn.utils.clip_grad_norm_(
                    optim_params, max_norm=max_grad_norm
                )
                grad_norms.append(grad_norm)

                # cycle
                lr_scheduler.step()
                optimizer.step()

            losses.append(loss.item())

            expected[expected == data_loader.dataset.token_to_id[PAD]] = -1     # copmarison between elements excpet PAD
            correct_symbols += torch.sum(sequence == expected[:, 1:], dim=(0, 1)).item()
            total_symbols += torch.sum(expected[:, 1:] != -1, dim=(0, 1)).item()

            pbar.update(curr_batch_size)

    expected = token_to_string(expected, data_loader)
    sequence = token_to_string(sequence, data_loader)
    print("-" * 10 + "GT ({})".format("train" if train else "valid"))
    print(*expected[:3], sep="\n")
    print("-" * 10 + "PR ({})".format("train" if train else "valid"))
    print(*sequence[:3], sep="\n")

    result = {
        "loss": np.mean(losses),
        "correct_symbols": correct_symbols,
        "total_symbols": total_symbols,
    }
    if train:
        # result["grad_norm"] = np.mean([tensor.cpu() for tensor in grad_norms])
        result["grad_norm"] = np.mean(grad_norms)

    return result


def main(config_file):
    """
    Train math formula recognition model
    """
    options = Flags(config_file).get()
    torch.manual_seed(options.seed)
    is_cuda = torch.cuda.is_available()
    hardware = "cuda" if is_cuda else "cpu"
    device = torch.device(hardware)
    print("--------------------------------")
    print("Running {} on device {}\n".format(options.network, device))

    # Print system environments
    num_gpus = torch.cuda.device_count()
    num_cpus = os.cpu_count()
    mem_size = virtual_memory().available // (1024 ** 3)
    print(
        "[+] System environments\n",
        "The number of gpus : {}\n".format(num_gpus),
        "The number of cpus : {}\n".format(num_cpus),
        "Memory Size : {}G\n".format(mem_size),
    )

    # Load checkpoint and print result
    checkpoint = (
        load_checkpoint(options.checkpoint, cuda=is_cuda)
        if options.checkpoint != ""
        else default_checkpoint
    )
    model_checkpoint = checkpoint["model"]
    if model_checkpoint:
        print(
            "[+] Checkpoint\n",
            "Resuming from epoch : {}\n".format(checkpoint["epoch"]),
            "Train Accuracy : {:.5f}\n".format(checkpoint["train_accuracy"][-1]),
            "Train Loss : {:.5f}\n".format(checkpoint["train_losses"][-1]),
            "Validation Accuracy : {:.5f}\n".format(
                checkpoint["validation_accuracy"][-1]
            ),
            "Validation Loss : {:.5f}\n".format(checkpoint["validation_losses"][-1]),
        )

    # Get data
    transformed = transforms.Compose(
        [
            # Resize so all images have the same size
            transforms.Resize((options.input_size.height, options.input_size.width)),
            transforms.ToTensor(),
        ]
    )
    (
        train_data_loader,
        validation_data_loader,
        train_dataset,
        valid_dataset,
    ) = dataset_loader(
        options.data.gt_paths,
        options.data.token_paths,
        options.data.dataset_proportions,
        options.data.split_proportions.train,
        options.data.split_proportions.valid,
        crop=options.data.crop,
        transform=transformed,
        batch_size=options.batch_size,
        num_workers=options.num_workers,
        rgb=options.data.rgb,
    )
    print(
        "[+] Data\n",
        "The number of train samples : {}\n".format(len(train_dataset)),
        "The number of validation samples : {}\n".format(len(valid_dataset)),
        "The number of classes : {}\n".format(len(train_dataset.token_to_id)),
    )

    # Get loss, model
    model = get_network(
        options.network,
        options,
        model_checkpoint,
        device,
        train_dataset,
    )
    model.train()
    criterion = model.criterion.to(device)
    enc_params_to_optimise = [
        param for param in model.encoder.parameters() if param.requires_grad
    ]
    dec_params_to_optimise = [
        param for param in model.decoder.parameters() if param.requires_grad
    ]
    params_to_optimise = [*enc_params_to_optimise, *dec_params_to_optimise]
    print(
        "[+] Network\n",
        "Type: {}\n".format(options.network),
        "Encoder parameters: {}\n".format(
            sum(p.numel() for p in enc_params_to_optimise),
        ),
        "Decoder parameters: {} \n".format(
            sum(p.numel() for p in dec_params_to_optimise),
        ),
    )

    # Get optimizer
    optimizer = get_optimizer(
        options.optimizer.optimizer,
        params_to_optimise,
        lr=options.optimizer.lr,
        weight_decay=options.optimizer.weight_decay,
    )
    optimizer_state = checkpoint.get("optimizer")
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
    for param_group in optimizer.param_groups:
        param_group["initial_lr"] = options.optimizer.lr
    if options.optimizer.is_cycle:
        cycle = len(train_data_loader) * options.num_epochs
        lr_scheduler = CircularLRBeta(
            optimizer, options.optimizer.lr, 10, 10, cycle, [0.95, 0.85]
        )
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=options.optimizer.lr_epochs,
            gamma=options.optimizer.lr_factor,
        )

    # Log
    if not os.path.exists(options.prefix):
        os.makedirs(options.prefix)
    log_file = open(os.path.join(options.prefix, "log.txt"), "w")
    shutil.copy(config_file, os.path.join(options.prefix, "train_config.yaml"))
    if options.print_epochs is None:
        options.print_epochs = options.num_epochs
    writer = init_tensorboard(name=options.prefix.strip("-"))
    start_epoch = checkpoint["epoch"]
    train_accuracy = checkpoint["train_accuracy"]
    train_losses = checkpoint["train_losses"]
    validation_accuracy = checkpoint["validation_accuracy"]
    validation_losses = checkpoint["validation_losses"]
    learning_rates = checkpoint["lr"]
    grad_norms = checkpoint["grad_norm"]

    # Train
    for epoch in range(options.num_epochs):
        start_time = time.time()

        epoch_text = "[{current:>{pad}}/{end}] Epoch {epoch}".format(
            current=epoch + 1,
            end=options.num_epochs,
            epoch=start_epoch + epoch + 1,
            pad=len(str(options.num_epochs)),
        )

        # Train
        train_result = run_epoch(
            train_data_loader,
            model,
            epoch_text,
            criterion,
            optimizer,
            lr_scheduler,
            options.teacher_forcing_ratio,
            options.max_grad_norm,
            device,
            train=True,
        )
        train_losses.append(train_result["loss"])
        grad_norms.append(train_result["grad_norm"])
        train_epoch_accuracy = (
            train_result["correct_symbols"] / train_result["total_symbols"]
        )
        train_accuracy.append(train_epoch_accuracy)
        epoch_lr = lr_scheduler.get_lr()  # cycle

        # Validation
        validation_result = run_epoch(
            validation_data_loader,
            model,
            epoch_text,
            criterion,
            optimizer,
            lr_scheduler,
            options.teacher_forcing_ratio,
            options.max_grad_norm,
            device,
            train=False,
        )
        validation_losses.append(validation_result["loss"])
        validation_epoch_accuracy = (
            validation_result["correct_symbols"] / validation_result["total_symbols"]
        )
        validation_accuracy.append(validation_epoch_accuracy)

        # Save checkpoint
        save_checkpoint(
            {
                "epoch": start_epoch + epoch + 1,
                "train_losses": train_losses,
                "train_accuracy": train_accuracy,
                "validation_losses": validation_losses,
                "validation_accuracy": validation_accuracy,
                "lr": learning_rates,
                "grad_norm": grad_norms,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            prefix=options.prefix,
        )

        # Summary
        elapsed_time = time.time() - start_time
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        if epoch % options.print_epochs == 0 or epoch == options.num_epochs - 1:
            output_string = (
                "{epoch_text}: "
                "Train Accuracy = {train_accuracy:.5f}, "
                "Train Loss = {train_loss:.5f}, "
                "Validation Accuracy = {validation_accuracy:.5f}, "
                "Validation Loss = {validation_loss:.5f}, "
                "lr = {lr} "
                "(time elapsed {time})"
            ).format(
                epoch_text=epoch_text,
                train_accuracy=train_epoch_accuracy,
                train_loss=train_result["loss"],
                validation_accuracy=validation_epoch_accuracy,
                validation_loss=validation_result["loss"],
                lr=epoch_lr,
                time=elapsed_time,
            )
            print(output_string)
            log_file.write(output_string + "\n")
            write_tensorboard(
                writer,
                start_epoch + epoch + 1,
                train_result["grad_norm"],
                train_result["loss"],
                train_epoch_accuracy,
                validation_result["loss"],
                validation_epoch_accuracy,
                model,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        dest="config_file",
        default="configs/SATRN.yaml",
        type=str,
        help="Path of configuration file",
    )
    parser = parser.parse_args()
    main(parser.config_file)

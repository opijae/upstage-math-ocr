import torch
import os
from train import id_to_string
from metrics import word_error_rate,sentence_acc
from checkpoint import load_checkpoint
from torchvision import transforms
from dataset import LoadDataset,collate_batch,START, PAD
from flags import Flags
from utils import get_network,get_optimizer
import csv
from torch.utils.data import DataLoader
import argparse
import random
from tqdm import tqdm

def main(config_file):
    options = Flags(config_file).get()

    torch.manual_seed(options.seed)
    random.seed(options.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    is_cuda = torch.cuda.is_available()
    hardware = "cuda" if is_cuda else "cpu"
    device = torch.device(hardware)
    print("--------------------------------")
    print("Running {} on device {}\n".format(options.network, device))


    checkpoint = load_checkpoint(options.checkpoint, cuda=is_cuda)
    model_checkpoint = checkpoint["model"]
    if model_checkpoint:
        print(
            "[+] Checkpoint\n",
            "Resuming from epoch : {}\n".format(checkpoint["epoch"]),
        )


    transformed = transforms.Compose(
        [
            transforms.Resize((options.input_size.height, options.input_size.width)),
            transforms.ToTensor(),
        ]
    )

    test_data=[]
    print(options.data)
    for path in options.data.test:
        root = os.path.join(os.path.dirname(path), "images")
        with open(path, "r") as fd:
            reader = csv.reader(fd, delimiter="\t")
            data = list(reader)
            random.shuffle(data)
        data = [[os.path.join(root, x[0]), x[1]] for x in data]
        test_data+=data
    test_dataset = LoadDataset(
        test_data, options.data.token_paths, crop=False, transform=transformed, rgb=options.data.rgb
    )
    test_data_loader= DataLoader(
        test_dataset,
        batch_size=options.batch_size,
        shuffle=False,
        num_workers=options.num_workers,
        collate_fn=collate_batch,
    )



    print(
        "[+] Data\n",
        "The number of test samples : {}\n".format(len(test_dataset)),
    )

    model = get_network(
        options.network,
        options,
        model_checkpoint,
        device,
        test_dataset,
    )
    model.eval()
    correct_symbols = 0
    total_symbols = 0
    wer=0
    num_wer=0
    sent_acc=0
    num_sent_acc=0
    for d in tqdm(test_data_loader):
        input = d["image"].to(device)
        expected = d["truth"]["encoded"].to(device)
        expected[expected == -1] = test_data_loader.dataset.token_to_id[PAD]
        output = model(input, expected, False, 0.0)
        decoded_values = output.transpose(1, 2)
        _, sequence = torch.topk(decoded_values, 1, dim=1)
        sequence = sequence.squeeze(1)
        expected[expected == test_data_loader.dataset.token_to_id[PAD]] = -1
        expected_str = id_to_string(expected, test_data_loader)
        sequence_str = id_to_string(sequence, test_data_loader)
        wer += word_error_rate(sequence_str, expected_str)
        num_wer += 1
        sent_acc += sentence_acc(sequence_str, expected_str)
        num_sent_acc += 1
        correct_symbols += torch.sum(sequence == expected[:, 1:], dim=(0, 1)).item()
        total_symbols += torch.sum(expected[:, 1:] != -1, dim=(0, 1)).item()
    print("wer",wer/num_wer)
    print("sentence acc",sent_acc/num_sent_acc)
    print("symbol acc",correct_symbols/total_symbols)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_file",
        dest="config_file",
        default="configs/SATRN_test.yaml",
        type=str,
        help="Path of configuration file",
    )
    # argument for predict vs. test
    parser = parser.parse_args()
    main(parser.config_file)

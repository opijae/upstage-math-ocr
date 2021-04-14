import csv
import os
import random
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

START = "<SOS>"
END = "<EOS>"
PAD = "<PAD>"
SPECIAL_TOKENS = [START, END, PAD]


# There are so many symbols (mostly escape sequences) that are in the test sets but not
# in the training set.
def remove_unknown_tokens(truth):
    # Remove \mathrm and \vtop are only present in the test sets, but not in the
    # training set. They are purely for formatting anyway.
    remaining_truth = truth.replace("\\mathrm", "")
    remaining_truth = remaining_truth.replace("\\vtop", "")
    # \; \! are spaces and only present in 2014's test set
    remaining_truth = remaining_truth.replace("\\;", " ")
    remaining_truth = remaining_truth.replace("\\!", " ")
    remaining_truth = remaining_truth.replace("\\ ", " ")
    # There's one occurrence of \dots in the 2013 test set, but it wasn't present in the
    # training set. It's either \ldots or \cdots in math mode, which are essentially
    # equivalent.
    remaining_truth = remaining_truth.replace("\\dots", "\\ldots")
    # Again, \lbrack and \rbrack where not present in the training set, but they render
    # similar to \left[ and \right] respectively.
    remaining_truth = remaining_truth.replace("\\lbrack", "\\left[")
    remaining_truth = remaining_truth.replace("\\rbrack", "\\right]")
    # Same story, where \mbox = \leavemode\hbox
    remaining_truth = remaining_truth.replace("\\hbox", "\\mbox")
    # There is no reason to use \lt or \gt instead of < and > in math mode. But the
    # training set does. They are not even LaTeX control sequences but are used in
    # MathJax (to prevent code injection).
    remaining_truth = remaining_truth.replace("<", "\\lt")
    remaining_truth = remaining_truth.replace(">", "\\gt")
    # \parallel renders to two vertical bars
    remaining_truth = remaining_truth.replace("\\parallel", "||")
    # Some capital letters are not in the training set...
    remaining_truth = remaining_truth.replace("O", "o")
    remaining_truth = remaining_truth.replace("W", "w")
    remaining_truth = remaining_truth.replace("\\Pi", "\\pi")
    return remaining_truth


# Rather ignorant way to encode the truth, but at least it works.
def encode_truth(truth, token_to_id):
    truth_tokens = []
    remaining_truth = remove_unknown_tokens(truth).strip()
    # TODO: we can simplify this
    while len(remaining_truth) > 0:
        try:
            matching_starts = [
                [i, len(tok)]
                for tok, i in token_to_id.items()
                if remaining_truth.startswith(tok)
            ]
            # Take the longest match
            index, tok_len = max(matching_starts, key=lambda match: match[1])
            truth_tokens.append(index)
            remaining_truth = remaining_truth[tok_len:].lstrip()
        except ValueError:
            raise Exception("Truth contains unknown token")
    return truth_tokens


def load_vocab(tokens_paths):
    tokens = []
    for tokens_file in tokens_paths:
        with open(tokens_file, "r") as fd:
            reader = fd.read()
            tokens += reader.split('\n')
    tokens.extend(SPECIAL_TOKENS)
    tokens = list(set(tokens))
    token_to_id = {tok: i for i, tok in enumerate(tokens)}
    id_to_token = {i: tok for i, tok in enumerate(tokens)}
    return token_to_id, id_to_token


def split_gt(groundtruth, validation_percent=0.2):
    root = os.path.dirname(groundtruth)
    with open(groundtruth, "r") as fd:
        reader = csv.reader(fd, delimiter="\t")
        data = list(reader)
        random.shuffle(data)
        validation_len = round(len(data) * validation_percent)
    data = [[os.path.join(root, x[0]), x[1]] for x in data]
    return data[validation_len:], data[:validation_len]


def collate_batch(data):
    max_len = max([len(d["truth"]["encoded"]) for d in data])
    # Padding with -1, will later be replaced with the PAD token
    padded_encoded = [
        d["truth"]["encoded"] + (max_len - len(d["truth"]["encoded"])) * [-1]
        for d in data
    ]
    return {
        "path": [d["path"] for d in data],
        "image": torch.stack([d["image"] for d in data], dim=0),
        "truth": {
            "text": [d["truth"]["text"] for d in data],
            "encoded": torch.tensor(padded_encoded),
            #"len_mask": 
        },
    }


class LoadDataset(Dataset):
    """Load Dataset"""

    def __init__(
        self,
        groundtruth,
        tokens_file,
        crop=False,
        transform=None,
    ):
        """
        Args:
            groundtruth (string): Path to ground truth TXT/TSV file
            tokens_file (string): Path to tokens TXT file
            ext (string): Extension of the input files
            crop (bool, optional): Crop images to their bounding boxes [Default: False]
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super(LoadDataset, self).__init__()
        self.crop = crop
        self.transform = transform
        self.token_to_id, self.id_to_token = load_vocab(tokens_file)
        self.data = [
            {
                "path": os.path.join('images', p),
                "truth": {
                    "text": truth,
                    "encoded": [
                        self.token_to_id[START],
                        *encode_truth(truth, self.token_to_id),
                        self.token_to_id[END],
                    ],
                },
            }
            for p, truth in groundtruth
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        image = Image.open(item["path"])
        # Remove alpha channel
        # image = image.convert("RGB")
        image = image.convert("L")

        if self.crop:
            # Image needs to be inverted because the bounding box cuts off black pixels,
            # not white ones.
            bounding_box = ImageOps.invert(image).getbbox()
            image = image.crop(bounding_box)

        if self.transform:
            image = self.transform(image)

        return {"path": item["path"], "truth": item["truth"], "image": image}


def dataset_loader(
    gt_paths,
    token_paths,
    dataset_proportions,
    train_proportion,
    valid_proportion,
    crop=False,
    transform=None,
    batch_size=16,
    num_workers=4,
):

    # Read data
    train_data, valid_data = [], []
    for path in gt_paths:
        train, valid = split_gt(path, valid_proportion)
        train_data += train
        valid_data += valid
    
    # Load data
    train_dataset = LoadDataset(train_data, token_paths, crop=crop, transform=transform)
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_batch,
    )

    valid_dataset = LoadDataset(valid_data, token_paths, crop=crop, transform=transform)
    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch,
    )
    
    return train_data_loader, valid_data_loader, train_dataset
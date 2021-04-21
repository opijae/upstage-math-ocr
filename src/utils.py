import torch.optim as optim

# from networks.crnn import Encoder, Decoder
from networks.crnn import CNN, RNN
from networks.transformer import (
    TransformerEncoderFor2DFeatures,
    AttentionDecoder,
    TransformerDecoder,
)
from dataset import START, PAD, END


def get_network(
    enc_type,
    dec_type,
    options,
    encoder_checkpoint,
    decoder_checkpoint,
    device,
    train_dataset,
):
    enc, dec = None, None

    if enc_type == "Transformer":
        enc = TransformerEncoderFor2DFeatures(
            input_size=options.data.rgb,
            hidden_dim=options.transformer.encoder_dim,
            filter_size=options.transformer.filter_size,
            head_num=8,
            layer_num=options.transformer.encoder_layers,
            dropout_rate=options.dropout_rate,
            checkpoint=encoder_checkpoint,
        ).to(device)
    elif enc_type == "CRNN":
        # enc = Encoder(
        #     img_channels=options.data.rgb,
        #     dropout_rate=options.dropout_rate,
        #     checkpoint=encoder_checkpoint,
        # ).to(device)
        enc = CNN(options.data.rgb, options.input_size.height)
    else:
        raise NotImplementedError

    if dec_type == "Transformer":
        dec = TransformerDecoder(
            len(train_dataset.id_to_token),
            src_dim=options.transformer.src_dim,
            hidden_dim=options.transformer.hidden_dim,
            filter_dim=options.transformer.filter_dim,
            head_num=options.transformer.head_num,
            dropout_rate=options.dropout_rate,
            pad_id=train_dataset.token_to_id[PAD],
            st_id=train_dataset.token_to_id[START],
            layer_num=options.transformer.dec_layers,
            checkpoint=decoder_checkpoint,
        ).to(device)
    elif dec_type == "Attention":
        dec = AttentionDecoder(
            len(train_dataset.id_to_token),
            src_dim=options.src_dim,
            embedding_dim=options.embedding_dim,
            hidden_dim=options.hidden_dim,
            pad_id=train_dataset.token_to_id[PAD],
            st_id=train_dataset.token_to_id[START],
            num_lstm_layers=1,
            checkpoint=decoder_checkpoint,
        ).to(device)
    elif dec_type == "CRNN":
        # dec = Decoder(
        #     len(train_dataset.id_to_token),
        #     low_res_shape,
        #     high_res_shape,
        #     checkpoint=decoder_checkpoint,
        #     device=device,
        # ).to(device)
        dec = RNN(options.hidden_dim, len(train_dataset.id_to_token)).to(device)
    else:
        raise NotImplementedError

    return enc, dec


def get_optimizer(optimizer, params, lr, weight_decay=None):
    if optimizer == "Adam":
        optimizer = optim.Adam(params, lr=lr)
    elif optimizer == "Adadelta":
        optim.Adadelta(params, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer


def get_loss_fn(decoder):
    if decoder == "CRNN":
        criterion = torch.nn.CTCLoss(
            ignore_index=train_data_loader.dataset.token_to_id[END]
        ).to(device)
    else:
        criterion = nn.CrossEntropyLoss(
            ignore_index=train_data_loader.dataset.token_to_id[PAD]
        ).to(device)
    return criterion
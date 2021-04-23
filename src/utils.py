import torch.optim as optim

# from networks.crnn import Encoder, Decoder
from networks.crnn import CNN, RNN
from networks.transformer import SATRN
from dataset import START, PAD, END


def get_network(
    model_type,
    FLAGS,
    model_checkpoint,
    device,
    train_dataset,
):
    """
    enc, dec = None, None

    if enc_type == "Transformer":
        enc = TransformerEncoderFor2DFeatures(
            input_size=FLAGS.data.rgb,
            hidden_dim=FLAGS.SATRN.encoder.hidden_dim,
            filter_size=FLAGS.SATRN.encoder.filter_dim,
            head_num=FLAGS.SATRN.encoder.head_num,
            layer_num=FLAGS.SATRN.encoder.layer_num,
            dropout_rate=FLAGS.dropout_rate,
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
            num_classes=len(train_dataset.id_to_token),
            src_dim=FLAGS.SATRN.decoder.src_dim,
            hidden_dim=FLAGS.SATRN.decoder.hidden_dim,
            filter_dim=FLAGS.SATRN.decoder.filter_dim,
            head_num=FLAGS.SATRN.decoder.head_num,
            dropout_rate=FLAGS.dropout_rate,
            pad_id=train_dataset.token_to_id[PAD],
            st_id=train_dataset.token_to_id[START],
            layer_num=FLAGS.SATRN.decoder.layer_num,
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
    """
    model = None

    if model_type == "SATRN":
        model = SATRN(FLAGS, train_dataset, model_checkpoint).to(device)
    elif model_type == "CRNN":
        model = CRNN()
    elif model_type == "FAN":
        model = FAN()

    return model


def get_optimizer(optimizer, params, lr, weight_decay=None):
    if optimizer == "Adam":
        optimizer = optim.Adam(params, lr=lr)
    elif optimizer == "Adadelta":
        optim.Adadelta(params, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer


def get_loss_fn(model):
    if decoder == "CRNN":
        criterion = torch.nn.CTCLoss(
            ignore_index=train_data_loader.dataset.token_to_id[END]
        ).to(device)
    else:
        criterion = nn.CrossEntropyLoss(
            ignore_index=train_data_loader.dataset.token_to_id[PAD]
        ).to(device)
    return criterion
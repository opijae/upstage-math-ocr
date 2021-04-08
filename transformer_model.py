import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Number of bottlenecks
num_bn = 3
# The depth is half of the actual values in the paper because bottleneck blocks
# are used which contain two convlutional layers
depth = 16
multi_block_depth = depth // 2
growth_rate = 24

n = 256
n_prime = 512
decoder_conv_filters = 256
gru_hidden_size = 256
embedding_dim = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ShallowCNN(nn.Module):
    """
    This is specialized to the math formula recognition task 
    - three convolutional layers to reduce the visual feature map size and to capture low-level visual features
    (128, 256) -> (8, 32) -> total 256 features
    - transformer layers cannot change the channel size, so it requires a wide feature dimension
    ***** this might be a point to be improved !!
    """
    def __init__(self, input_channel, output_channel=256, dropout_rate=0.2):
        super(ShallowCNN, self).__init__()
        self.output_channel = [int(output_channel//8), int(output_channel//4), int(output_channel//2)]  # [32, 64, 128]
        self.ConvNet = nn.Sequential(
                nn.Conv2d(input_channel, self.output_channel[0], 3, 2, 1),
                nn.BatchNorm2d(self.output_channel[0]), nn.ReLU(True), # 32x (64x128)
                nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 2, 1),
                nn.BatchNorm2d(self.output_channel[1]), nn.ReLU(True), # 64 x (32x64)
                nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 2, 1),
                nn.BatchNorm2d(self.output_channel[2]), nn.ReLU(True) # 128 x (16x32)
            )

        # self.output_channel = [int(output_channel//4), int(output_channel//2)]  # [32, 64, 128]
        # self.ConvNet = nn.Sequential(
        #         nn.Conv2d(input_channel, self.output_channel[0], 3, 2, 1),
        #         nn.BatchNorm2d(self.output_channel[0]), nn.ReLU(True), # 32x (64x128)
        #         nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 2, 1),
        #         nn.BatchNorm2d(self.output_channel[1]), nn.ReLU(True) # 64 x (32x64)
        #     )


    def forward(self, input):
        out = self.ConvNet(input) # 128 x (16x32)
        
        # concat adjacent features (height reduction)
        b, c, h, w = out.size()
        out = out.view(b, c, h//2, 2, w).transpose(2, 3).contiguous()
        out = out.view(b, 2*c, h//2, w).contiguous()

        return out # 256 x (8x32)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask=mask, value=float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, head_num=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.in_channels = in_channels
        self.head_dim = in_channels // head_num
        self.head_num = head_num

        self.q_linear = nn.Linear(in_channels, self.head_num * self.head_dim)
        self.k_linear = nn.Linear(in_channels, self.head_num * self.head_dim)
        self.v_linear = nn.Linear(in_channels, self.head_num * self.head_dim)
        self.attention = ScaledDotProductAttention(temperature=(self.head_num * self.head_dim) ** 0.5, dropout=dropout)
        self.out_linear = nn.Linear(self.head_num * self.head_dim, in_channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        b, q_len, k_len, v_len = q.size(0), q.size(1), k.size(1), v.size(1)
        q = self.q_linear(q).view(b, q_len, self.head_num, self.head_dim).transpose(1, 2)
        k = self.k_linear(k).view(b, k_len, self.head_num, self.head_dim).transpose(1, 2)
        v = self.v_linear(v).view(b, v_len, self.head_num, self.head_dim).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        out, attn = self.attention(q, k, v, mask=mask)
        out = out.transpose(1, 2).contiguous().view(b, q_len, self.head_num * self.head_dim)
        out = self.out_linear(out)
        out = self.dropout(out)

        return out

class Feedforward(nn.Module):
    def __init__(self, filter_size=2048, hidden_dim=512, dropout=0.1):
        super(Feedforward, self).__init__()

        self.layers = nn.Sequential(
                nn.Linear(hidden_dim, filter_size, True), nn.ReLU(True),
                nn.Dropout(p=dropout),
                nn.Linear(filter_size, hidden_dim, True), nn.ReLU(True),
                nn.Dropout(p=dropout),
            )
    def forward(self, input):
        return self.layers(input)


class TransformerEncoderLayer(nn.Module):
    
    def __init__(self, input_size, filter_size, head_num, dropout_rate=0.2):
        super(TransformerEncoderLayer, self).__init__()

        self.attention_layer = MultiHeadAttention(in_channels=input_size, head_num=head_num, dropout=dropout_rate)
        self.attention_norm = nn.LayerNorm(normalized_shape=input_size)
        self.feedforward_layer = Feedforward(filter_size=filter_size, hidden_dim=input_size)
        self.feedforward_norm = nn.LayerNorm(normalized_shape=input_size)

    def forward(self, input):

        out = self.attention_norm(input)
        out = self.attention_layer(out, out, out)
        out = self.feedforward_norm(out)
        out = self.feedforward_layer(out)

        return out

class PositionalEncoding2D(nn.Module):

    def __init__(self, in_channels, max_h=16, max_w=64, dropout=0.1):
        super(PositionalEncoding2D, self).__init__()

        self.h_position_encoder = self.generate_encoder(in_channels//2, max_h)
        self.w_position_encoder = self.generate_encoder(in_channels//2, max_w)

        self.h_linear = nn.Linear(in_channels//2, in_channels//2)
        self.w_linear = nn.Linear(in_channels//2, in_channels//2)

        self.dropout = nn.Dropout(p=dropout)

    def generate_encoder(self, in_channels, max_len):
        pos = torch.arange(max_len).float().unsqueeze(1)
        i = torch.arange(in_channels).float().unsqueeze(0)
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / in_channels)
        position_encoder = pos * angle_rates
        position_encoder[:, 0::2] = torch.sin(position_encoder[:, 0::2])
        position_encoder[:, 1::2] = torch.cos(position_encoder[:, 1::2])
        return position_encoder # (Max_len, In_channel)

    def forward(self, input):
        ### Require DEBUG
        b, c, h, w = input.size()
        h_pos_encoding = self.h_position_encoder[:h, :].unsqueeze(1).to(input.get_device())
        h_pos_encoding = self.h_linear(h_pos_encoding) # [H, 1, D] 

        w_pos_encoding = self.w_position_encoder[:w, :].unsqueeze(0).to(input.get_device())
        w_pos_encoding = self.w_linear(w_pos_encoding) # [1, W, D]

        h_pos_encoding = h_pos_encoding.expand(-1, w, -1)
        w_pos_encoding = w_pos_encoding.expand(h, -1, -1)

        pos_encoding = torch.cat([h_pos_encoding, w_pos_encoding], dim=2) # [H, W, 2*D]

        pos_encoding = pos_encoding.permute(2, 0, 1) # [2*D, H, W]

        out = input + pos_encoding.unsqueeze(0) 
        out = self.dropout(out)

        return out

class TransformerEncoderFor2DFeatures(nn.Module):
    """
    Transformer Encoder for Image
    1) ShallowCNN : low-level visual feature identification and dimension reduction
    2) Positional Encoding : adding positional information to the visual features
    3) Transformer Encoders : self-attention layers for the 2D feature maps
    """

    def __init__(self, input_size, hidden_dim, filter_size, head_num, layer_num, dropout_rate=0.1, checkpoint=None):
        super(TransformerEncoderFor2DFeatures, self).__init__()

        self.shallow_cnn = ShallowCNN(input_size, output_channel=hidden_dim, dropout_rate=dropout_rate)
        self.positional_encoding = PositionalEncoding2D(hidden_dim)
        self.attention_layers = nn.ModuleList(
            [ TransformerEncoderLayer(hidden_dim, filter_size, head_num, dropout_rate) for _ in range(layer_num)]
            )
        if checkpoint is not None:
            self.load_state_dict(checkpoint)

    def forward(self, input):

        out = self.shallow_cnn(input) #[b, c, h*, w*] (h*=8, w*=32)
        out = self.positional_encoding(out) #[b, c, h*, w*]

        # flatten
        b, c, h, w = out.size()
        out = out.view(b, c, h*w).transpose(1, 2) # [b, h* x w* (L), c]

        for layer in self.attention_layers:
            out = layer(out)
        return out


class AttentionCell(nn.Module):

    def __init__(self, src_dim, hidden_dim, embedding_dim, num_lstm_layers=1):
        super(AttentionCell, self).__init__()
        self.num_lstm_layers = num_lstm_layers

        self.i2h = nn.Linear(src_dim, hidden_dim, bias=False)
        self.h2h = nn.Linear(hidden_dim, hidden_dim)  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_dim, 1, bias=False)
        if num_lstm_layers == 1:
            self.rnn = nn.LSTMCell(src_dim + embedding_dim, hidden_dim)
        else:
            self.rnn = nn.ModuleList([nn.LSTMCell(src_dim + embedding_dim, hidden_dim)] + [nn.LSTMCell(hidden_dim, hidden_dim) for _ in range(num_lstm_layers-1)])

        self.hidden_dim = hidden_dim

    def forward(self, prev_hidden, src, tgt):

        src_features = self.i2h(src) # [b, L, c] (image features)
        if self.num_lstm_layers == 1:
            prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        else:
            prev_hidden_proj = self.h2h(prev_hidden[-1][0]).unsqueeze(1)
        attention_logit = self.score(torch.tanh(src_features + prev_hidden_proj)) # [b, L, 1] 
        alpha = F.softmax(attention_logit, dim=1) # [b, L, 1] 
        context = torch.bmm(alpha.permute(0, 2, 1), src).squeeze(1)  # [b, c]

        concat_context = torch.cat([context, tgt], 1)  # [b, c+e]

        if self.num_lstm_layers == 1:
            cur_hidden = self.rnn(concat_context, prev_hidden)
        else:
            cur_hidden = []
            for i, layer in enumerate(self.rnn):
                if i == 0:
                    concat_context = layer(concat_context, prev_hidden[i])
                else:
                    concat_context = layer(concat_context[0], prev_hidden[i])
                cur_hidden.append(concat_context)

        return cur_hidden, alpha

class AttentionDecoder(nn.Module):

    def __init__(self, num_classes, src_dim, embedding_dim, hidden_dim, num_lstm_layers=1, checkpoint=None):
        super(AttentionDecoder, self).__init__()

        self.embedding = nn.Embedding(num_classes+1, embedding_dim)
        self.attention_cell = AttentionCell(src_dim, hidden_dim, embedding_dim, num_lstm_layers)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_lstm_layers = num_lstm_layers
        self.generator = nn.Linear(hidden_dim, num_classes)

        if checkpoint is not None:
            self.load_state_dict(checkpoint)

    def forward(self, src, text, is_train=True, batch_max_length=50):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x contextual_feature_channels]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        batch_size = src.size(0)
        num_steps = batch_max_length - 1 # +1 for [s] at end of sentence.

        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_dim).fill_(0).to(device)
        if self.num_lstm_layers == 1:
            hidden = (torch.FloatTensor(batch_size, self.hidden_dim).fill_(0).to(device),
                      torch.FloatTensor(batch_size, self.hidden_dim).fill_(0).to(device))
        else:
            hidden = [(torch.FloatTensor(batch_size, self.hidden_dim).fill_(0).to(device),
                      torch.FloatTensor(batch_size, self.hidden_dim).fill_(0).to(device)) for _ in range(self.num_lstm_layers)]

        if is_train:
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                embedd = self.embedding( text[:, i] )
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                hidden, alpha = self.attention_cell(hidden, src, embedd)
                if self.num_lstm_layers == 1:
                    output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)
                else:
                    output_hiddens[:, i, :] = hidden[-1][0]
            probs = self.generator(output_hiddens)

        else:
            targets = torch.LongTensor(batch_size).fill_(0).to(device)  # [GO] token
            probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(device)

            for i in range(num_steps):
                embedd = self.embedding(targets)
                hidden, alpha = self.attention_cell(hidden, src, embedd)
                if self.num_lstm_layers == 1:
                    probs_step = self.generator(hidden[0])
                else:
                    probs_step = self.generator(hidden[-1][0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input

        return probs  # batch_size x num_steps x num_classes

class Decoder(nn.Module):
    """Decoder

    GRU based Decoder which attends to the low- and high-resolution annotations to
    create a LaTeX string.
    """

    def __init__(
        self,
        num_classes,
        res_shape,
        hidden_size=256,
        embedding_dim=256,
        checkpoint=None,
        device=device,
    ):
        """
        Args:
            num_classes (int): Number of symbol classes
            low_res_shape ((int, int, int)): Shape of the low resolution annotations
                i.e. (C, W, H)
            high_res_shape ((int, int, int)): Shape of the high resolution annotations
                i.e. (C_prime, 2W, 2H)
            hidden_size (int, optional): Hidden size of the GRU [Default: 256]
            embedding_dim (int, optional): Dimension of the embedding [Default: 256]
            checkpoint (dict, optional): State dictionary to be loaded
            device (torch.device, optional): Device for the tensors
        """
        super(Decoder, self).__init__()
        
        context_size = res_shape[0]
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.gru1 = nn.GRU(
            input_size=embedding_dim, hidden_size=hidden_size, batch_first=True
        )
        self.gru2 = nn.GRU(
            input_size=context_size, hidden_size=hidden_size, batch_first=True
        )
        # L = H * W
        res_attn_size = res_shape[1]*res_shape[2]

        self.coverage_attn_low = CoverageAttention(
            context_size,
            hidden_size,
            attn_size=res_attn_size,
            kernel_size=(11, 11),
            padding=5,
            device=device,
        )
        
        self.W_o = nn.Parameter(torch.empty((num_classes, embedding_dim // 2)))
        self.W_s = nn.Parameter(torch.empty((embedding_dim, hidden_size)))
        self.W_c = nn.Parameter(torch.empty((embedding_dim, context_size)))
        self.U_pred = nn.Parameter(torch.empty((n_prime, n)))
        self.maxout = Maxout(2)
        self.hidden_size = hidden_size
        nn.init.xavier_normal_(self.W_o)
        nn.init.xavier_normal_(self.W_s)
        nn.init.xavier_normal_(self.W_c)
        nn.init.xavier_normal_(self.U_pred)

        if checkpoint is not None:
            self.load_state_dict(checkpoint)

    def init_hidden(self, batch_size):
        return torch.zeros((1, batch_size, self.hidden_size))

    def reset(self, batch_size):
        self.coverage_attn_low.reset_alpha(batch_size)
        self.coverage_attn_high.reset_alpha(batch_size)

    # Unsqueeze and squeeze are used to add and remove the seq_len dimension,
    # which is always 1 since only the previous symbol is provided, not a sequence.
    # The inputs that are multiplied by the weights are transposed to get
    # (m x batch_size) instead of (batch_size x m). The result of the
    # multiplication is tranposed back.
    def forward(self, x, hidden, low_res, high_res):
        embedded = self.embedding(x)
        pred, _ = self.gru1(embedded, hidden)
        # u_pred is computed here instead of in the coverage attention, because the
        # weight U_pred is shared and the coverage attention does not use pred for
        # anything else. This avoids computing it twice.
        u_pred = torch.matmul(self.U_pred, pred.squeeze(1).t()).t()
        context_low = self.coverage_attn_low(low_res, u_pred)
        context_high = self.coverage_attn_high(high_res, u_pred)
        context = torch.cat((context_low, context_high), dim=1)
        new_hidden, _ = self.gru2(context.unsqueeze(1), pred.transpose(0, 1))
        w_s = torch.matmul(self.W_s, new_hidden.squeeze(1).t()).t()
        w_c = torch.matmul(self.W_c, context.t()).t()
        out = embedded.squeeze(1) + w_s + w_c
        out = self.maxout(out)
        out = torch.matmul(self.W_o, out.t()).t()
        return out, new_hidden.transpose(0, 1)

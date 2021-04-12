import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random

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

from dataset import START, PAD

class BottleneckBlock(nn.Module):
    """
    Dense Bottleneck Block

    It contains two convolutional layers, a 1x1 and a 3x3.
    """

    def __init__(self, input_size, growth_rate, dropout_rate=0.2):
        """
        Args:
            input_size (int): Number of channels of the input
            growth_rate (int): Number of new features being added. That is the ouput
                size of the last convolutional layer.
            dropout_rate (float, optional): Probability of dropout [Default: 0.2]
        """
        super(BottleneckBlock, self).__init__()
        inter_size = num_bn * growth_rate
        self.norm1 = nn.BatchNorm2d(input_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            input_size, inter_size, kernel_size=1, stride=1, bias=False
        )
        self.norm2 = nn.BatchNorm2d(inter_size)
        self.conv2 = nn.Conv2d(
            inter_size, growth_rate, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.conv1(self.relu(self.norm1(x)))
        out = self.conv2(self.relu(self.norm2(out)))
        out = self.dropout(out)
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    """
    Transition Block

    A transition layer reduces the number of feature maps in-between two bottleneck
    blocks.
    """

    def __init__(self, input_size, output_size):
        """
        Args:
            input_size (int): Number of channels of the input
            output_size (int): Number of channels of the output
        """
        super(TransitionBlock, self).__init__()
        self.norm = nn.BatchNorm2d(input_size)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            input_size, output_size, kernel_size=1, stride=1, bias=False
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(self.relu(self.norm(x)))
        return self.pool(out)


class DenseBlock(nn.Module):
    """
    Dense block

    A dense block stacks several bottleneck blocks.
    """

    def __init__(self, input_size, growth_rate, depth, dropout_rate=0.2):
        """
        Args:
            input_size (int): Number of channels of the input
            growth_rate (int): Number of new features being added per bottleneck block
            depth (int): Number of bottleneck blocks
            dropout_rate (float, optional): Probability of dropout [Default: 0.2]
        """
        super(DenseBlock, self).__init__()
        layers = [
            BottleneckBlock(
                input_size + i * growth_rate, growth_rate, dropout_rate=dropout_rate
            )
            for i in range(depth)
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class DeepCNN128(nn.Module):
    """
    This is specialized to the math formula recognition task 
    - three convolutional layers to reduce the visual feature map size and to capture low-level visual features
    (128, 256) -> (8, 32) -> total 256 features
    - transformer layers cannot change the channel size, so it requires a wide feature dimension
    ***** this might be a point to be improved !!
    """
    def __init__(self, input_channel, num_in_features, output_channel=256, dropout_rate=0.2):
        super(DeepCNN128, self).__init__()
        self.conv0 = nn.Conv2d(
            input_channel, # 3
            num_in_features, # 32
            kernel_size=7,stride=2,padding=3,bias=False,
        )
        self.norm0 = nn.BatchNorm2d(num_in_features)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2) # 1/4 (128, 128) -> (32, 32)
        num_features = num_in_features
       
        self.block1 = DenseBlock(
            num_features, # 32
            growth_rate=16, # 32 + growth_rate(16)*depth(14) -> 256
            depth=14, # 16?
            dropout_rate=0.2,
        )
        num_features = 256 #num_features + depth * growth_rate
        self.trans1 = TransitionBlock(num_features, num_features // 2) # 16 x 16
        num_features = num_features // 2
        self.block2 = DenseBlock(
            num_features, # 128
            growth_rate=16, #16
            depth=8, #8
            dropout_rate=0.2,
        )
        num_features = 256
        self.trans2_norm = nn.BatchNorm2d(num_features)
        self.trans2_relu = nn.ReLU(inplace=True)
        self.trans2_conv = nn.Conv2d(
            num_features, num_features // 2, kernel_size=1, stride=1, bias=False #128
        )

    def forward(self, input):
        out = self.conv0(input) # (H, V, )
        out = self.relu(self.norm0(out))
        out = self.max_pool(out)
        out = self.block1(out)
        out = self.trans1(out)
        out = self.block2(out)
        out_before_trans2 = self.trans2_relu(self.trans2_norm(out))
        out_A = self.trans2_conv(out_before_trans2)

        return out_A # 128 x (16x16)

class DeepCNN300(nn.Module):
    """
    This is specialized to the math formula recognition task 
    - three convolutional layers to reduce the visual feature map size and to capture low-level visual features
    (128, 256) -> (8, 32) -> total 256 features
    - transformer layers cannot change the channel size, so it requires a wide feature dimension
    ***** this might be a point to be improved !!
    """
    def __init__(self, input_channel, num_in_features, output_channel=256, dropout_rate=0.2):
        super(DeepCNN300, self).__init__()
        self.conv0 = nn.Conv2d(
            input_channel, # 3
            num_in_features, # 48
            kernel_size=7,stride=2,padding=3,bias=False,
        )
        self.norm0 = nn.BatchNorm2d(num_in_features)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2) # 1/4 (128, 128) -> (32, 32)
        num_features = num_in_features
       
        self.block1 = DenseBlock(
            num_features, # 48
            growth_rate=growth_rate, # 48 + growth_rate(16)*depth(24) -> 432
            depth=depth, # 16?
            dropout_rate=0.2,
        )
        num_features = num_features + depth * growth_rate
        self.trans1 = TransitionBlock(num_features, num_features // 2) # 16 x 16
        num_features = num_features // 2
        self.block2 = DenseBlock(
            num_features, # 128
            growth_rate=growth_rate, #16
            depth=depth, #8
            dropout_rate=0.2,
        )
        num_features = num_features + depth * growth_rate
        self.trans2_norm = nn.BatchNorm2d(num_features)
        self.trans2_relu = nn.ReLU(inplace=True)
        self.trans2_conv = nn.Conv2d(
            num_features, num_features // 2, kernel_size=1, stride=1, bias=False #128
        )

    def forward(self, input):
        out = self.conv0(input) # (H, V, )
        out = self.relu(self.norm0(out))
        out = self.max_pool(out)
        out = self.block1(out)
        out = self.trans1(out)
        out = self.block2(out)
        out_before_trans2 = self.trans2_relu(self.trans2_norm(out))
        out_A = self.trans2_conv(out_before_trans2)

        return out_A # 128 x (16x16)

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
        # self.output_channel = [int(output_channel//8), int(output_channel//4), int(output_channel//2)]  # [32, 64, 128]
        # self.ConvNet = nn.Sequential(
        #         nn.Conv2d(input_channel, self.output_channel[0], 3, 2, 1),
        #         nn.BatchNorm2d(self.output_channel[0]), nn.ReLU(True), # 32x (64x128)
        #         nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 2, 1),
        #         nn.BatchNorm2d(self.output_channel[1]), nn.ReLU(True), # 64 x (32x64)
        #         nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 2, 1),
        #         nn.BatchNorm2d(self.output_channel[2]), nn.ReLU(True) # 128 x (16x32)
        #     )

        # self.output_channel = [int(output_channel//4), int(output_channel//2)]  # [32, 64, 128]
        # self.ConvNet = nn.Sequential(
        #         nn.Conv2d(input_channel, self.output_channel[0], 3, 2, 1),
        #         nn.BatchNorm2d(self.output_channel[0]), nn.ReLU(True), # 32x (64x128)
        #         nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 2, 1),
        #         nn.BatchNorm2d(self.output_channel[1]), nn.ReLU(True) # 64 x (32x64)
        #     )

        # Like Origin
        self.output_channel = [int(output_channel//4), int(output_channel//2)]  # [32, 64, 128]
        self.ConvNet = nn.Sequential(
                nn.Conv2d(input_channel, self.output_channel[0], 7, 2, 3),
                nn.BatchNorm2d(self.output_channel[0]), nn.ReLU(True), # 32x (64x128)
                nn.Conv2d(self.output_channel[0], self.output_channel[1], 7, 2, 3),
                nn.BatchNorm2d(self.output_channel[1]), nn.ReLU(True), # 64 x (32x64)
                # nn.Conv2d(self.output_channel[1], self.output_channel[2], 7, 2, 3),
                # nn.BatchNorm2d(self.output_channel[2]), nn.ReLU(True) # 64 x (32x64)
            )


    def forward(self, input):
        out = self.ConvNet(input) # 128 x (16x32)
        
        # concat adjacent features (height reduction)
        b, c, h, w = out.size()
        out = out.view(b, c, h//2, 2, w).transpose(2, 3).contiguous()
        out = out.view(b, 2*c, h//2, w).contiguous()

        return out # 256 x (16x64)


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
    def __init__(self, q_channels, k_channels, head_num=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.q_channels = q_channels
        self.k_channels = k_channels
        self.head_dim = q_channels // head_num
        self.head_num = head_num

        self.q_linear = nn.Linear(q_channels, self.head_num * self.head_dim)
        self.k_linear = nn.Linear(k_channels, self.head_num * self.head_dim)
        self.v_linear = nn.Linear(k_channels, self.head_num * self.head_dim)
        self.attention = ScaledDotProductAttention(temperature=(self.head_num * self.head_dim) ** 0.5, dropout=dropout)
        self.out_linear = nn.Linear(self.head_num * self.head_dim, q_channels)
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

        self.attention_layer = MultiHeadAttention(q_channels=input_size, k_channels=input_size, head_num=head_num, dropout=dropout_rate)
        self.attention_norm = nn.LayerNorm(normalized_shape=input_size)
        self.feedforward_layer = Feedforward(filter_size=filter_size, hidden_dim=input_size)
        self.feedforward_norm = nn.LayerNorm(normalized_shape=input_size)

    def forward(self, input):

        att = self.attention_layer(input, input, input)
        out = self.attention_norm(att + input)

        ff = self.feedforward_layer(out)
        out = self.feedforward_norm(ff + out)
        return out

class PositionalEncoding2D(nn.Module):

    def __init__(self, in_channels, max_h=64, max_w=128, dropout=0.1):
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

        #self.shallow_cnn = ShallowCNN(input_size, output_channel=hidden_dim, dropout_rate=dropout_rate)
        #self.shallow_cnn = DeepCNN128(input_size, num_in_features=32, output_channel=hidden_dim, dropout_rate=dropout_rate)
        self.shallow_cnn = DeepCNN300(input_size, num_in_features=48, output_channel=hidden_dim, dropout_rate=dropout_rate)
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

class TransformerDecoderLayer(nn.Module):
    def __init__(self, input_size, src_size, filter_size, head_num, dropout_rate=0.2):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attention_layer = MultiHeadAttention(q_channels=input_size, k_channels=input_size, head_num=head_num, dropout=dropout_rate)
        self.self_attention_norm = nn.LayerNorm(normalized_shape=input_size)

        self.attention_layer = MultiHeadAttention(q_channels=input_size, k_channels=src_size, head_num=head_num, dropout=dropout_rate)
        self.attention_norm = nn.LayerNorm(normalized_shape=input_size)

        self.feedforward_layer = Feedforward(filter_size=filter_size, hidden_dim=input_size)
        self.feedforward_norm = nn.LayerNorm(normalized_shape=input_size)

    def forward(self, tgt, tgt_prev, src, tgt_mask):

        if tgt_prev == None: # Train
            att = self.self_attention_layer(tgt, tgt, tgt, tgt_mask)
            out = self.self_attention_norm(att + tgt )

            att = self.attention_layer(tgt, src, src)
            out = self.attention_norm(att + out )

            ff = self.feedforward_layer(out)
            out = self.feedforward_norm(ff + out )
        else: 
            tgt_prev = torch.cat([tgt_prev, tgt], 1)
            att = self.self_attention_layer(tgt, tgt_prev, tgt_prev, tgt_mask)
            out = self.self_attention_norm(att + tgt )

            att = self.attention_layer(tgt, src, src)
            out = self.attention_norm(att + out )

            ff = self.feedforward_layer(out)
            out = self.feedforward_norm(ff + out )
        return out

class PositionEncoder1D(nn.Module):
    def __init__(self, in_channels, max_len=500, dropout=0.1):
        super(PositionEncoder1D, self).__init__()

        self.position_encoder = self.generate_encoder(in_channels, max_len)
        self.position_encoder = self.position_encoder.unsqueeze(0)
        self.dropout = nn.Dropout(p=dropout)

    def generate_encoder(self, in_channels, max_len):
        pos = torch.arange(max_len).float().unsqueeze(1)

        i = torch.arange(in_channels).float().unsqueeze(0)
        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / in_channels)

        position_encoder = pos * angle_rates
        position_encoder[:, 0::2] = torch.sin(position_encoder[:, 0::2])
        position_encoder[:, 1::2] = torch.cos(position_encoder[:, 1::2])

        return position_encoder

    def forward(self, x, point=-1):
        if point == -1:
            out = x + self.position_encoder[:, :x.size(1), :].to(x.get_device())
            out = self.dropout(out)
        else:
            out = x + self.position_encoder[:, point, :].unsqueeze(1).to(x.get_device())
        return out

class TransformerDecoder(nn.Module):
    def __init__(self, num_classes, src_dim, hidden_dim, filter_dim, head_num, dropout_rate, pad_id, st_id, layer_num=1, checkpoint=None):
        super(TransformerDecoder, self).__init__()

        self.embedding = nn.Embedding(num_classes+1, hidden_dim)
        self.hidden_dim = hidden_dim
        self.filter_dim = filter_dim
        self.num_classes = num_classes
        self.layer_num = layer_num

        self.pos_encoder = PositionEncoder1D(in_channels=hidden_dim, dropout=dropout_rate)

        self.attention_layers = nn.ModuleList(
            [ TransformerDecoderLayer(hidden_dim, src_dim, filter_dim, head_num, dropout_rate) for _ in range(layer_num)]
            )
        self.generator = nn.Linear(hidden_dim, num_classes)

        self.pad_id = pad_id
        self.st_id = st_id

        if checkpoint is not None:
            self.load_state_dict(checkpoint)

    def pad_mask(self, text):
        pad_mask = (text == self.pad_id)
        pad_mask[:, 0] = False
        pad_mask = pad_mask.unsqueeze(1)

        return pad_mask

    def order_mask(self, length):
        order_mask = torch.triu(torch.ones(length,length), diagonal=1).bool()
        order_mask = order_mask.unsqueeze(0).to(device)
        return order_mask

    def text_embedding(self, texts):
        tgt = self.embedding(texts)
        tgt *= math.sqrt(tgt.size(2))

        return tgt

    def forward(self, src, text, is_train=True, batch_max_length=50, teacher_forcing_ratio=1.0):

        if is_train and random.random() < teacher_forcing_ratio:
            tgt = self.text_embedding(text)
            tgt = self.pos_encoder(tgt)
            tgt_mask = (self.pad_mask(text) | self.order_mask(text.size(1)))
            for layer in self.attention_layers:
                tgt = layer(tgt, None, src, tgt_mask)
            out = self.generator(tgt)
        else:
            out = []
            num_steps = batch_max_length - 1
            target = torch.LongTensor(src.size(0)).fill_(self.st_id).to(device)
            
            features = [None]*self.layer_num

            for t in range(num_steps):
                target = target.unsqueeze(1)
                tgt = self.text_embedding(target)
                tgt = self.pos_encoder(tgt, point=t)
                tgt_mask = self.order_mask(t+1)
                tgt_mask = tgt_mask[:, -1].unsqueeze(1) # [1, (l+1)]
                for l, layer in enumerate(self.attention_layers):
                    tgt = layer(tgt, features[l], src, tgt_mask)
                    features[l] = tgt if features[l] == None else torch.cat([features[l],tgt],1)

                _out = self.generator(tgt) # [b, 1, c]
                target = torch.argmax(_out[:, -1:, :], dim=-1)
                target = target.squeeze()

                out.append(_out)
            out = torch.stack(out, dim=1).to(device)
            out = out.squeeze(2)

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

    def __init__(self, num_classes, src_dim, embedding_dim, hidden_dim, pad_id, st_id, num_lstm_layers=1, checkpoint=None):
        super(AttentionDecoder, self).__init__()

        self.embedding = nn.Embedding(num_classes+1, embedding_dim)
        self.attention_cell = AttentionCell(src_dim, hidden_dim, embedding_dim, num_lstm_layers)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_lstm_layers = num_lstm_layers
        self.generator = nn.Linear(hidden_dim, num_classes) 
        self.pad_id = pad_id
        self.st_id = st_id

        if checkpoint is not None:
            self.load_state_dict(checkpoint)


    def forward(self, src, text, is_train=True, batch_max_length=50, teacher_forcing_ratio=1.0):
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

        if is_train and random.random() < teacher_forcing_ratio:
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
            targets = torch.LongTensor(batch_size).fill_(self.st_id).to(device)  # [GO] token
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


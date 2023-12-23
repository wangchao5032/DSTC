#coding:utf-8
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import os
from utils import constants


class StackingGRUCell(nn.Module):
    """
    Multi-layer CRU Cell
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(StackingGRUCell, self).__init__()
        self.num_layers = num_layers
        self.grus = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        self.grus.append(nn.GRUCell(input_size, hidden_size))
        for i in range(1, num_layers):
            self.grus.append(nn.GRUCell(hidden_size, hidden_size))

    def forward(self, input, h0):
        """
        Input:
        input (batch, input_size): input tensor
        h0 (num_layers, batch, hidden_size): initial hidden state
        ---
        Output:
        output (batch, hidden_size): the final layer output tensor
        hn (num_layers, batch, hidden_size): the hidden state of each layer
        """
        hn = []
        output = input
        for i, gru in enumerate(self.grus):
            hn_i = gru(output, h0[i])
            hn.append(hn_i)
            if i != self.num_layers - 1:
                output = self.dropout(hn_i)
            else:
                output = hn_i
        hn = torch.stack(hn)
        return output, hn


class StackingGRUCell_without_dropout(nn.Module):
    """
    Multi-layer CRU Cell 没有dropout的版本
    """
    def __init__(self, input_size, hidden_size, num_layers):
        super(StackingGRUCell_without_dropout, self).__init__()
        self.num_layers = num_layers
        self.grus = nn.ModuleList()

        self.grus.append(nn.GRUCell(input_size, hidden_size))
        for i in range(1, num_layers):
            self.grus.append(nn.GRUCell(hidden_size, hidden_size))

    def forward(self, input, h0):
        """
        Input:
        input (batch, input_size): input tensor
        h0 (num_layers, batch, hidden_size): initial hidden state
        ---
        Output:
        output (batch, hidden_size): the final layer output tensor
        hn (num_layers, batch, hidden_size): the hidden state of each layer
        """
        hn = []
        output = input
        for i, gru in enumerate(self.grus):
            hn_i = gru(output, h0[i])
            hn.append(hn_i)
            if i != self.num_layers - 1:
                output = hn_i
            else:
                output = hn_i
        hn = torch.stack(hn)
        return output, hn


class GlobalAttention(nn.Module):
    """
    $$a = \sigma((W_1 q)H)$$
    $$c = \tanh(W_2 [a H, q])$$
    """
    def __init__(self, hidden_size):
        super(GlobalAttention, self).__init__()
        self.L1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.L2 = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def forward(self, q, H):
        """
        Input:
        q (batch, hidden_size): query
        H (batch, seq_len, hidden_size): context
        ---
        Output:
        c (batch, hidden_size)
        """
        # (batch, hidden_size) => (batch, hidden_size, 1)
        q1 = self.L1(q).unsqueeze(2)
        # (batch, seq_len)
        a = torch.bmm(H, q1).squeeze(2)
        a = self.softmax(a)
        # (batch, seq_len) => (batch, 1, seq_len)
        a = a.unsqueeze(1)
        # (batch, hidden_size)
        c = torch.bmm(a, H).squeeze(1)
        # (batch, hidden_size * 2)
        c = torch.cat([c, q], 1)
        return self.tanh(self.L2(c))


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout,
                       bidirectional, embedding):
        """
        embedding (vocab_size, input_size): pretrained embedding
        """
        super(Encoder, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        assert hidden_size % self.num_directions == 0
        self.hidden_size = hidden_size // self.num_directions # 如果是双向的隐藏层分半，分别用于正向RNN和逆向RNN
        self.num_layers = num_layers

        self.embedding = embedding
        self.rnn = nn.GRU(input_size, self.hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          dropout=dropout)

    def forward(self, input, lengths, h0=None):
        """
        Input:
        input (seq_len, batch): padded sequence tensor
        lengths (1, batch): sequence lengths
        h0 (num_layers*num_directions, batch, hidden_size): initial hidden state
        ---
        Output:
        hn (num_layers*num_directions, batch, hidden_size):
            the hidden state of each layer
        output (seq_len, batch, hidden_size*num_directions): output tensor
        """
        # (seq_len, batch) => (seq_len, batch, input_size)
        # Encoder的前向传播过程包括embedding+rnn，rnn实际上是GRU
        # 嵌入层，转成256维的向量
        # batch个轨迹，对轨迹上的点都映射为256维的向量
        embed = self.embedding(input) #传入batch个句子，每个句子共50个单词，如（50，128）。embed(50,batch,256)每个单词用256向量表示
        lengths = lengths.data.view(-1).tolist()
        if lengths is not None:
            embed = pack_padded_sequence(embed, lengths)
        output, hn = self.rnn(embed, h0)
        if lengths is not None:
            output = pad_packed_sequence(output)[0] # (50,128,256)
        return hn, output


class Encoder_without_dropout(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                       bidirectional, embedding):
        """
        embedding (vocab_size, input_size): pretrained embedding 无dropout的版本
        """
        super(Encoder_without_dropout, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        assert hidden_size % self.num_directions == 0
        self.hidden_size = hidden_size // self.num_directions
        self.num_layers = num_layers

        self.embedding = embedding
        self.rnn = nn.GRU(input_size, self.hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          )

    def forward(self, input, lengths, h0=None):
        """
        Input:
        input (seq_len, batch): padded sequence tensor
        lengths (1, batch): sequence lengths
        h0 (num_layers*num_directions, batch, hidden_size): initial hidden state
        ---
        Output:
        hn (num_layers*num_directions, batch, hidden_size):
            the hidden state of each layer
        output (seq_len, batch, hidden_size*num_directions): output tensor
        """
        # (seq_len, batch) => (seq_len, batch, input_size)
        # Encoder的前向传播过程包括embedding+rnn，rnn实际上是GRU
        with torch.autograd.set_detect_anomaly(True):
            embed = self.embedding(input) #传入batch个句子，每个句子共50个单词，如（50，128）。embed(50,batch,256)每个单词用256向量表示
            lengths = lengths.data.view(-1).tolist()
            if lengths is not None:
                 # 按长度压缩，把seq_len, batch 合并为一个，对应每个轨迹每个点的向量,即变为[所有轨迹总长度, inputsize]
                embed = pack_padded_sequence(embed, lengths)
            output, hn = self.rnn(embed, h0)
            if lengths is not None:
                output = pad_packed_sequence(output)[0] # (50,128,256)
            return hn, output


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, embedding):
        super(Decoder, self).__init__()
        self.embedding = embedding
        self.rnn = StackingGRUCell(input_size, hidden_size, num_layers,
                                   dropout)
        self.attention = GlobalAttention(hidden_size)
        self.dropout = nn.Dropout(dropout) #减少过拟合现象，神经元有dropout的几率被关闭
        self.num_layers = num_layers

    def forward(self, input, h, H, use_attention=True):
        """
        Input:
        input (seq_len, batch): padded sequence tensor; input是trg数据
        h (num_layers, batch, hidden_size): input hidden state; h是encoder输出的hn
        H (seq_len, batch, hidden_size): the context used in attention mechanism
            which is the output of encoder                         # H是上下文，是encoder的输出
        use_attention: If True then we use attention
        ---
        Output:
        output (seq_len, batch, hidden_size)
        h (num_layers, batch, hidden_size): output hidden state,
            h may serve as input hidden state for the next iteration,
            especially when we feed the word one by one (i.e., seq_len=1)
            such as in translation
        """
        assert input.dim() == 2, "The input should be of (seq_len, batch)"
        # (seq_len, batch) => (seq_len, batch, input_size)
        # 首先都通过嵌入层， 原始轨迹也处理
        embed = self.embedding(input) #把trg数据也用词向量表示
        output = []
        # split along the sequence length dimension
        for e in embed.split(1):
            e = e.squeeze(0) # (1, batch, input_size) => (batch, input_size)
            o, h = self.rnn(e, h)
            if use_attention:
                o = self.attention(o, H.transpose(0, 1))
            o = self.dropout(o)
            output.append(o)
        output = torch.stack(output)
        return output, h


class Decoder_without_dropout(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, embedding):
        super(Decoder_without_dropout, self).__init__()
        self.embedding = embedding
        self.rnn = StackingGRUCell_without_dropout(input_size, hidden_size, num_layers)
        self.attention = GlobalAttention(hidden_size)
        self.num_layers = num_layers

    def forward(self, input, h, H, use_attention=True):
        """
        Input:
        input (seq_len, batch): padded sequence tensor; input是trg数据
        h (num_layers, batch, hidden_size): input hidden state; h是encoder输出的hn
        H (seq_len, batch, hidden_size): the context used in attention mechanism
            which is the output of encoder                         # H是上下文，是encoder的输出
        use_attention: If True then we use attention
        ---
        Output:
        output (seq_len, batch, hidden_size)
        h (num_layers, batch, hidden_size): output hidden state,
            h may serve as input hidden state for the next iteration,
            especially when we feed the word one by one (i.e., seq_len=1)
            such as in translation
        """
        assert input.dim() == 2, "The input should be of (seq_len, batch)"
        # (seq_len, batch) => (seq_len, batch, input_size)
        embed = self.embedding(input) #把trg数据也用词向量表示
        output = []
        # split along the sequence length dimension
        for e in embed.split(1):
            e = e.squeeze(0) # (1, batch, input_size) => (batch, input_size)
            o, h = self.rnn(e, h)
            if use_attention:
                o = self.attention(o, H.transpose(0, 1))
            output.append(o)
        output = torch.stack(output)
        return output, h


class EncoderDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout, bidirectional):
        super(EncoderDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        ## the embedding shared by encoder and decoder
        # # 长度不一样，所以可以用PAD填充，网络遇到这种符号不必理会
        self.embedding = nn.Embedding(vocab_size, embedding_size,
                                      padding_idx=constants.PAD)
        #vocab_size是字典的大小，embedding_size是每一个向量的大小；padding_idx 遇到此下标时用0填充
        self.encoder = Encoder(embedding_size, hidden_size, num_layers,
                               dropout, bidirectional, self.embedding)
        self.decoder = Decoder(embedding_size, hidden_size, num_layers,
                               dropout, self.embedding)
        self.num_layers = num_layers

    def load_pretrained_embedding(path):
        if os.path.isfile(path):
            w = torch.load(path)
            self.embedding.weight.data.copy_(w)

    def encoder_hn2decoder_h0(self, h):
        """
        Input:
        h (num_layers * num_directions, batch, hidden_size): encoder output hn
        ---
        Output:
        h (num_layers, batch, hidden_size * num_directions): decoder input h0
        """
        if self.encoder.num_directions == 2:
            num_layers, batch, hidden_size = h.size(0)//2, h.size(1), h.size(2)
            return h.view(num_layers, 2, batch, hidden_size)\
                    .transpose(1, 2).contiguous()\
                    .view(num_layers, batch, hidden_size * 2)
        else:
            return h

    def forward(self, src, lengths, trg):
        """
        Input:
        src (src_seq_len, batch): source tensor
        lengths (1, batch): source sequence lengths
        trg (trg_seq_len, batch): target tensor, the `seq_len` in trg is not
            necessarily the same as that in src
        ---
        Output:
        output (trg_seq_len, batch, hidden_size)
        """
        # 最后一步的hidden_state和每个时间步得到的hidden_state
        encoder_hn, H = self.encoder(src, lengths)
        # 编码器的输出作为解码器的输入，先处理一下
        decoder_h0 = self.encoder_hn2decoder_h0(encoder_hn)
        ## for target we feed the range [BOS:EOS-1] into decoder
        output, decoder_hn = self.decoder(trg[:-1], decoder_h0, H)
        return output,decoder_h0


class EncoderDecoder_without_dropout(nn.Module):
    def __init__(self, vocab_size, embedding_size,
                       hidden_size, num_layers, bidirectional):
        super(EncoderDecoder_without_dropout, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        ## the embedding shared by encoder and decoder
        self.embedding = nn.Embedding(vocab_size, embedding_size,
                                      padding_idx=constants.PAD)
        #vocab_size是字典的大小，embedding_size是每一个向量的大小；padding_idx 遇到此下标时用0填充
        self.encoder = Encoder_without_dropout(embedding_size, hidden_size, num_layers, bidirectional, self.embedding)
        self.decoder = Decoder_without_dropout(embedding_size, hidden_size, num_layers, self.embedding)
        self.num_layers = num_layers

    def load_pretrained_embedding(path):
        if os.path.isfile(path):
            w = torch.load(path)
            self.embedding.weight.data.copy_(w)

    def encoder_hn2decoder_h0(self, h):
        """
        Input:
        h (num_layers * num_directions, batch, hidden_size): encoder output hn
        ---
        Output:
        h (num_layers, batch, hidden_size * num_directions): decoder input h0
        """
        if self.encoder.num_directions == 2:
            num_layers, batch, hidden_size = h.size(0)//2, h.size(1), h.size(2)
            return h.view(num_layers, 2, batch, hidden_size)\
                    .transpose(1, 2).contiguous()\
                    .view(num_layers, batch, hidden_size * 2)
        else:
            return h

    def forward(self, src, lengths, trg):
        """
        Input:
        src (src_seq_len, batch): source tensor
        lengths (1, batch): source sequence lengths
        trg (trg_seq_len, batch): target tensor, the `seq_len` in trg is not
            necessarily the same as that in src
        ---
        Output:
        output (trg_seq_len, batch, hidden_size)
        """
        encoder_hn, H = self.encoder(src, lengths)
        decoder_h0 = self.encoder_hn2decoder_h0(encoder_hn)
        ## for target we feed the range [BOS:EOS-1] into decoder
        output, decoder_hn = self.decoder(trg[:-1], decoder_h0, H)
        return output,decoder_h0

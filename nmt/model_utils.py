import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hid_dim,
        n_layers,
        n_heads,
        pf_dim,
        encoder_layer,
        self_attention_layer,
        positionwise_feedforward_layer,
        dropout,
        device,
    ):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(1000, hid_dim)
        self.layers = nn.ModuleList(
            [
                encoder_layer(
                    hid_dim,
                    n_heads,
                    pf_dim,
                    self_attention_layer,
                    positionwise_feedforward_layer,
                    dropout,
                    device,
                )
                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        batch_size = src.shape[0]
        src_len = src.shape[1]
        pos = (
            torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        )
        src = self.dropout(
            (self.tok_embedding(src) * self.scale) + self.pos_embedding(pos)
        )
        for layer in self.layers:
            src = layer(src, src_mask)
        return src


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hid_dim,
        n_heads,
        pf_dim,
        self_attention_layer,
        positionwise_feedforward_layer,
        dropout,
        device,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = self_attention_layer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = positionwise_feedforward_layer(
            hid_dim, pf_dim, dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        _src, _ = self.self_attention(src, src, src, src_mask)
        # dropout, residual connection and layer norm
        src = self.layer_norm(src + self.dropout(_src))
        # positionwise feedforward, dropout, residual and layer norm
        src = self.layer_norm(src + self.dropout(self.positionwise_feedforward(src)))
        return src


class SelfAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        if hid_dim % n_heads != 0:
            raise AssertionError("hid_dim % n_head != 0")

        self.n_heads = n_heads
        self.hid_dim = hid_dim
        self.head_dim = hid_dim // n_heads

        self.fc = nn.Linear(hid_dim, hid_dim)
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc(x)
        return x, attention


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim,
        hid_dim,
        n_layers,
        n_heads,
        pf_dim,
        decoder_layer,
        self_attention_layer,
        positionwise_feedforward_layer,
        dropout,
        device,
    ):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(1000, hid_dim)

        self.layers = nn.ModuleList(
            [
                decoder_layer(
                    hid_dim,
                    n_heads,
                    pf_dim,
                    self_attention_layer,
                    positionwise_feedforward_layer,
                    dropout,
                    device,
                )
                for _ in range(n_layers)
            ]
        )

        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        pos = (
            torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        )
        trg = self.dropout(
            (self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos)
        )
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        output = self.fc_out(trg)
        return output, attention


class DecoderLayer(nn.Module):
    def __init__(
        self,
        hid_dim,
        n_heads,
        pf_dim,
        self_attention_layer,
        positionwise_feedforward_layer,
        dropout,
        device,
    ):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = self_attention_layer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = self_attention_layer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = positionwise_feedforward_layer(
            hid_dim, pf_dim, dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        # dropout, residual connection and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        # dropout, residual connection and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))
        # positionwise feedforward, dropout, residual and layer norm
        trg = self.layer_norm(trg + self.dropout(self.positionwise_feedforward(trg)))
        return trg, attention


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, trg_sos_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(
            torch.ones((trg_len, trg_len), device=self.device)
        ).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output, attention


def load_model(ckpt_path, SRC_vocab, TRG_vocab):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = len(SRC_vocab)
    output_dim = len(TRG_vocab)
    hid_dim = 128
    encoder_layers = 3
    decoder_layers = 3
    encoder_heads = 8
    decoder_heads = 8
    encoder_pf_dimension = 256
    decoder_pf_dimension = 256
    encoder_dropout = 0.1
    decoder_dropout = 0.1

    enc = Encoder(
        input_dim,
        hid_dim,
        encoder_layers,
        encoder_heads,
        encoder_pf_dimension,
        EncoderLayer,
        SelfAttentionLayer,
        PositionwiseFeedforwardLayer,
        encoder_dropout,
        device,
    )

    dec = Decoder(
        output_dim,
        hid_dim,
        decoder_layers,
        decoder_heads,
        decoder_pf_dimension,
        DecoderLayer,
        SelfAttentionLayer,
        PositionwiseFeedforwardLayer,
        decoder_dropout,
        device,
    )

    src_pad_idx = SRC_vocab.stoi["<pad>"]
    trg_pad_idx = TRG_vocab.stoi["<pad>"]
    trg_sds_idx = TRG_vocab.stoi["<sos>"]

    model = Seq2Seq(enc, dec, src_pad_idx, trg_pad_idx, trg_sds_idx, device).to(device)
    load_ckpt(model=model, ckpt_path=ckpt_path, device=device)
    print(f"The model has {count_parameters(model):,} trainable parameters")
    return model


def load_ckpt(model, ckpt_path, device=torch.device("cpu")):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from .data_utils import load_vocab

    _SRC_vocab, _TRG_vocab = load_vocab("models/SRC_vocab.pkl", "models/TRG_vocab.pkl")
    _model = load_model("models/model.pt", _SRC_vocab, _TRG_vocab)

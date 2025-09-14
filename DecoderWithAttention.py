import torch.nn as nn
import torch
from Attention import Attention


class DecoderWithAttention(nn.Module):
    def __init__(
        self,
        attention_dim,
        embed_dim,
        decoder_dim,
        vocab_size,
        encoder_dim=512,
        dropout=0.5,
    ):
        super(DecoderWithAttention, self).__init__()
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.encoder_dim = encoder_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.vocab_size = vocab_size
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(2) * encoder_out.size(3)
        encoder_out = encoder_out.permute(0, 2, 3, 1)  # -> (B, H, W, C)
        encoder_out = encoder_out.view(batch_size, num_pixels, self.encoder_dim)
        vocab_size = self.vocab_size

        embeddings = self.embedding(encoded_captions)
        h, c = self.init_hidden_state(encoder_out)

        decode_lengths = [c - 1 for c in caption_lengths]
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(
            encoder_out.device
        )

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, _ = self.attention(
                encoder_out[:batch_size_t], h[:batch_size_t]
            )
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            lstm_input = torch.cat(
                (embeddings[:batch_size_t, t, :], attention_weighted_encoding), dim=1
            )
            h, c = self.decode_step(lstm_input, (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds

        return predictions, encoded_captions, decode_lengths

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    "位置编码"
    def __init__(self, d_model, dropout, device, max_len=1024):  # 将 max_len 从 512 改为 1024
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化 Shape 为 (max_len, d_model) 的 PE (positional encoding)
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len).unsqueeze(1).to(device)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).to(device)

        pe[:, 0::2] = torch.sin(position.float() * div_term)

        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position.float() * div_term)
        else:
            pe[:, 1::2] = torch.cos(position.float() * div_term[:-1] if len(div_term) > d_model // 2 else div_term)

        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"Sequence length {x.size(1)} exceeds positional encoding max_len {self.pe.size(1)}.")
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


class TranslationModel(nn.Module):
    def __init__(self, d_model, src_vocab, tgt_vocab, max_seq_length, device, dropout=0.1):
        super(TranslationModel, self).__init__()
        self.device = device

        self.src_embedding = nn.Embedding(len(src_vocab), d_model, padding_idx=2)
        self.tgt_embedding = nn.Embedding(len(tgt_vocab), d_model, padding_idx=2)
        self.positional_encoding = PositionalEncoding(d_model, dropout, device, max_len=1024)  # 统一 max_len

        self.transformer = nn.Transformer(d_model, dropout=dropout, batch_first=True)
        self.predictor = nn.Linear(d_model, len(tgt_vocab))

    def forward(self, src, tgt):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(-1)).to(self.device)
        src_key_padding_mask = TranslationModel.get_key_padding_mask(src)
        tgt_key_padding_mask = TranslationModel.get_key_padding_mask(tgt)

        src = self.src_embedding(src)
        tgt = self.tgt_embedding(tgt)

        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        out = self.transformer(src, tgt,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)
        return out

    @staticmethod
    def get_key_padding_mask(tokens, pad_idx=2):
        return tokens == pad_idx


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    from data.cmn_hun import TranslationDataset

    dataset = TranslationDataset(r"D:\2025-up\MT\TransformerTurnDistillation\data\cmn-eng\cmn.txt")
    model = TranslationModel(512, dataset.en_vocab, dataset.zh_vocab, 1024, device)
    model = model.to(device)

    en = "hello world"
    input = torch.tensor([0] + dataset.en_vocab(dataset.en_tokenizer(en)) + [1]).unsqueeze(0).to(device)

    zh = "你"
    output = torch.tensor([0] + dataset.zh_vocab(dataset.zh_tokenizer(zh))).unsqueeze(0).to(device)

    result = model(input, output)
    print(result)

    predict = model.predictor(result[:, -1])
    print(predict)

    y = torch.argmax(predict, dim=1).cpu().item()
    print(dataset.zh_vocab.lookup_tokens([y]))

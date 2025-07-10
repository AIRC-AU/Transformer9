import torch
from pathlib import Path
from data.cmn_eng import TranslationDataset

# 工作目录，缓存文件和模型会放在该目录下
base_dir = r"D:\QQ\TransformerTurnDistillation\train_process\transformer-cnn"
work_dir = Path(base_dir)
model_dir = Path(base_dir + "/transformer_checkpoints")
device = torch.device('cpu')  # 强制使用CPU
data_dir = "data/cmn-eng/cmn.txt"

dataset = TranslationDataset(data_dir)
max_seq_length = 42


def translate(src: str):
    # 加载模型时明确指定加载到CPU
    model = torch.load(model_dir / 'model_45000.pt', map_location=torch.device('cpu'))

    # 确保模型在CPU上
    model.to(device)

    # 将模型中的device属性设置为CPU（如果模型有这个属性的话）
    if hasattr(model, 'device'):
        model.device = device

    model.eval()

    src = torch.tensor([0] + dataset.en_vocab(dataset.en_tokenizer(src)) + [1], dtype=torch.long).unsqueeze(0).to(
        device)

    # 从 <bos> 开始
    tgt = torch.tensor([[0]], dtype=torch.long, device=device)

    for i in range(max_seq_length):
        # 确保所有输入都在CPU上
        src = src.to(device)
        tgt = tgt.to(device)

        out = model(src, tgt)
        predict = model.predictor(out[:, -1])
        y = torch.argmax(predict, dim=1)
        tgt = torch.cat([tgt, y.unsqueeze(0)], dim=1)
        if y.item() == 1:  # <eos>
            break

    tgt_tokens = tgt.squeeze().tolist()
    tgt_sentence = " ".join(dataset.zh_vocab.lookup_tokens(tgt_tokens))
    tgt_sentence = tgt_sentence.replace("<s>", "").replace("</s>", "").strip()
    return tgt_sentence


if __name__ == "__main__":
    print(translate("The train will probably arrive at the station before noon."))
    print(translate("like "))
    print(translate("我是帅哥，我爱学习，我爱说实话"))
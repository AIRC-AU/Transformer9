import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torchtext.data import get_tokenizer
import jieba
from torchtext.vocab import build_vocab_from_iterator
import zhconv
import re


class TranslationDataset(Dataset):
    def __init__(self, filepath, use_cache=True):
        self.row_count = self.get_row_count(filepath)
        self.use_cache = use_cache

        # 加载词典和token
        self.zh_vocab = self.get_zh_vocab(filepath)
        self.hun_vocab = self.get_hun_vocab(filepath)
        self.zh_tokens = self.load_tokens(filepath, self.zh_tokenizer, self.zh_vocab, "构建英文tokens", 'zh')
        self.hun_tokens = self.load_tokens(filepath, self.hun_tokenizer, self.hun_vocab, "构建匈牙利语tokens", 'hun')

    def __getitem__(self, index):
        return self.zh_tokens[index], self.hun_tokens[index]

    def __len__(self):
        return self.row_count

    def load_tokens(self, filepath, tokenizer, vocab, desc, lang):
        dir_path = os.path.dirname(filepath)
        cache_file = os.path.join(dir_path, f"tokens_list_{lang}.pt")
        if self.use_cache and os.path.exists(cache_file):
            print(f"正在加载缓存文件[{cache_file}]，请稍候...")
            return torch.load(cache_file, map_location="cpu")

        tokens_list = []
        with open(filepath, encoding='utf-8') as f:
            for line in tqdm(f, desc=desc, total=self.row_count):
                sentence = line.strip().split('\t')
                if len(sentence) >= 2:
                    if lang == 'zh':
                        # 使用英语文本作为源语言（在实际应用中应该是中文）
                        # 注意：这是一个简化的实现，实际应用中需要真实的中文数据
                        text = sentence[0].casefold()  # 英语文本作为"中文"的占位符
                    else:  # hun
                        text = sentence[1].lower()  # 匈牙利语文本

                    tokens = tokenizer(text)
                    token_indices = [vocab[token] for token in tokens]
                    token_tensor = torch.tensor([vocab["<s>"]] + token_indices + [vocab["</s>"]])
                    tokens_list.append(token_tensor)

        if self.use_cache:
            torch.save(tokens_list, cache_file)
        return tokens_list

    def get_row_count(self, filepath):
        count = 0
        for _ in open(filepath, encoding='utf-8'):
            count += 1
        return count

    def zh_tokenizer(self, line):
        # 注意：由于我们使用英语文本作为"中文"的占位符，这里使用英语分词器
        # 在实际应用中，如果有真实的中文数据，应该使用jieba分词器：
        # return list(jieba.cut(line))
        tokenizer = get_tokenizer('basic_english')
        return tokenizer(line)

    def hun_tokenizer(self, line):
        # 匈牙利语分词器 - 简单的基于空格和标点的分词
        # 匈牙利语是一种黏着语，可能需要更复杂的分词逻辑
        line = re.sub(r'[^\w\s]', ' ', line)  # 替换标点符号为空格
        tokens = line.split()
        return [token for token in tokens if token.strip()]

    def get_zh_vocab(self, filepath):
        def yield_zh_tokens():
            with open(filepath, encoding='utf-8') as f:
                print("---开始构建源语言词典（英语作为中文占位符）---")
                for line in tqdm(f, desc="构建源语言词典", total=self.row_count):
                    sentence = line.split('\t')
                    if len(sentence) >= 1:
                        # 使用英语文本作为源语言（在实际应用中应该是中文）
                        text = sentence[0].casefold()
                        yield self.zh_tokenizer(text)

        dir_path = os.path.dirname(filepath)
        zh_vocab_file = os.path.join(dir_path, "vocab_zh.pt")
        if self.use_cache and os.path.exists(zh_vocab_file):
            zh_vocab = torch.load(zh_vocab_file, map_location="cpu")
        else:
            zh_vocab = build_vocab_from_iterator(
                yield_zh_tokens(),
                min_freq=2,
                specials=["<s>", "</s>", "<pad>", "<unk>"]
            )
            zh_vocab.set_default_index(zh_vocab["<unk>"])
            if self.use_cache:
                torch.save(zh_vocab, zh_vocab_file)
        return zh_vocab

    def get_hun_vocab(self, filepath):
        def yield_hun_tokens():
            with open(filepath, encoding='utf-8') as f:
                print("---开始构建匈牙利语词典---")
                for line in tqdm(f, desc="构建匈牙利语词典", total=self.row_count):
                    sentence = line.split('\t')
                    if len(sentence) >= 2:
                        hungarian = sentence[1].lower()
                        yield self.hun_tokenizer(hungarian)

        dir_path = os.path.dirname(filepath)
        hun_vocab_file = os.path.join(dir_path, "vocab_hun.pt")
        if self.use_cache and os.path.exists(hun_vocab_file):
            hun_vocab = torch.load(hun_vocab_file, map_location="cpu")
        else:
            hun_vocab = build_vocab_from_iterator(
                yield_hun_tokens(),
                min_freq=1,
                specials=["<s>", "</s>", "<pad>", "<unk>"]
            )
            hun_vocab.set_default_index(hun_vocab["<unk>"])
            if self.use_cache:
                torch.save(hun_vocab, hun_vocab_file)
        return hun_vocab


if __name__ == '__main__':
    dataset = TranslationDataset(r"D:\2025-up\MT\TransformerTurnDistillation\data\hun-eng\hun.txt")
    print("句子数量为:", dataset.row_count)
    print("英文词典大小:", len(dataset.zh_vocab))
    print("匈牙利语词典大小:", len(dataset.hun_vocab))
    # 输出中文词典前10个索引
    print("英文词典前10个:", dict((i, dataset.zh_vocab.lookup_token(i)) for i in range(10)))
    # 输出匈牙利语词典前10个索引
    print("匈牙利语词典前10个:", dict((i, dataset.hun_vocab.lookup_token(i)) for i in range(10)))
    # 输出前5个句子对应的字典索引编号
    print("前5个英文tokens:", dict((i, dataset.zh_tokens[i]) for i in range(5)))
    print("前5个匈牙利语tokens:", dict((i, dataset.hun_tokens[i]) for i in range(5)))

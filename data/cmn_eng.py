import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from torchtext.data import get_tokenizer
import jieba
from torchtext.vocab import build_vocab_from_iterator
import zhconv


class TranslationDataset(Dataset):
    def __init__(self, filepath, use_cache=True):
        self.row_count = self.get_row_count(filepath)
        self.tokenizer = get_tokenizer('basic_english')
        self.use_cache = use_cache

        # 加载词典和token
        self.en_vocab = self.get_en_vocab(filepath)
        self.zh_vocab = self.get_zh_vocab(filepath)
        self.en_tokens = self.load_tokens(filepath, self.en_tokenizer, self.en_vocab, "构建英文tokens", 'en')
        self.zh_tokens = self.load_tokens(filepath, self.zh_tokenizer, self.zh_vocab, "构建中文tokens", 'zh')

    def __getitem__(self, index):
        return self.en_tokens[index], self.zh_tokens[index]

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
                if (lang == 'en' and len(sentence) >= 1) or (lang != 'en' and len(sentence) >= 2):
                    if lang == 'en':
                        text = sentence[0].casefold()
                    else:
                        text = zhconv.convert(sentence[1], 'zh-cn')
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

    def en_tokenizer(self, line):
        return self.tokenizer(line)

    def zh_tokenizer(self, line):
        return list(jieba.cut(line))

    def get_en_vocab(self, filepath):
        def yield_en_tokens():
            with open(filepath, encoding='utf-8') as f:
                print("---开始构建英文词典---")
                for line in tqdm(f, desc="构建英文词典", total=self.row_count):
                    sentence = line.split('\t')
                    if len(sentence) >= 1:
                        english = sentence[0]
                        yield self.en_tokenizer(english)

        dir_path = os.path.dirname(filepath)
        en_vocab_file = os.path.join(dir_path, "vocab_en.pt")
        if self.use_cache and os.path.exists(en_vocab_file):
            en_vocab = torch.load(en_vocab_file, map_location="cpu")
        else:
            en_vocab = build_vocab_from_iterator(
                yield_en_tokens(),
                min_freq=2,
                specials=["<s>", "</s>", "<pad>", "<unk>"]
            )
            en_vocab.set_default_index(en_vocab["<unk>"])
            if self.use_cache:
                torch.save(en_vocab, en_vocab_file)
        return en_vocab

    def get_zh_vocab(self, filepath):
        def yield_zh_tokens():
            with open(filepath, encoding='utf-8') as f:
                print("---开始构建中文词典---")
                for line in tqdm(f, desc="构建中文词典", total=self.row_count):
                    sentence = line.split('\t')
                    if len(sentence) >= 2:
                        chinese = zhconv.convert(sentence[1], 'zh-cn')
                        yield self.zh_tokenizer(chinese)

        dir_path = os.path.dirname(filepath)
        zh_vocab_file = os.path.join(dir_path, "vocab_zh.pt")
        if self.use_cache and os.path.exists(zh_vocab_file):
            zh_vocab = torch.load(zh_vocab_file, map_location="cpu")
        else:
            zh_vocab = build_vocab_from_iterator(
                yield_zh_tokens(),
                min_freq=1,
                specials=["<s>", "</s>", "<pad>", "<unk>"]
            )
            zh_vocab.set_default_index(zh_vocab["<unk>"])
            if self.use_cache:
                torch.save(zh_vocab, zh_vocab_file)
        return zh_vocab

if __name__ == '__main__':
    dataset = TranslationDataset(r"D:\2025-up\MT\TransformerTurnDistillation\data\cmn-eng\cmn.txt")
    print("句子数量为:", dataset.row_count)  # 29668
    print(dataset.en_tokenizer("I'm an English tokenizer."))  # ['i', "'m", 'an', 'english', 'tokenizer', '.']
    print("英文词典大小:", len(dataset.en_vocab))  # 4459
    # 输出英文词典前10个索引
    print(dict((i, dataset.en_vocab.lookup_token(i)) for i in range(10)))
    print("中文词典大小:", len(dataset.zh_vocab))  # 12519
    # 输出中文词典前10个索引
    print(dict((i, dataset.zh_vocab.lookup_token(i)) for i in range(10)))
    # 输出英文前10个句子对应的字典索引编号
    print(dict((i, dataset.en_tokens[i]) for i in range(10)))
    # 输出中文前10个句子对应的字典索引编号
    print(dict((i, dataset.zh_tokens[i]) for i in range(10)))
    print(dataset.en_vocab(['hello', 'tom']))  # 该词在词典中的索引[1706, 13]
    print(dataset.zh_vocab(['你', '好', '汤', '姆']))  # 该词在词典中的索引[8, 38, 2380, 3]
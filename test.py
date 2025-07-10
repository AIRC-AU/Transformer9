import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import jieba
import zhconv
from tqdm import tqdm

# 测试英文分词器
print(get_tokenizer('basic_english')('How are you?'))

# 测试中文分词
print(list(jieba.cut('今天你还好吗？')))







import torch

class CharTokenizer:
    """字符级分词器，自动处理未知字符（如中文等非词表字符会被跳过）"""
    def __init__(self, text):
        if text:
            chars = sorted(list(set(text)))
        else:
            chars = []
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s):
        """将字符串编码为索引列表，忽略未知字符"""
        ids = []
        for ch in s:
            if ch in self.stoi:
                ids.append(self.stoi[ch])
        return ids

    def decode(self, ids):
        """将索引列表解码为字符串"""
        return ''.join([self.itos.get(i, '') for i in ids])


def get_batch(data, block_size, batch_size, device):
    """从数据集中随机采样一个batch"""
    if data is None or len(data) == 0:
        # 返回空张量避免崩溃
        x = torch.zeros((batch_size, block_size), dtype=torch.long, device=device)
        y = torch.zeros((batch_size, block_size), dtype=torch.long, device=device)
        return x, y

    max_start = max(1, len(data) - block_size)
    if max_start <= 0:
        # 数据长度不足，用现有数据填充
        x = torch.zeros((batch_size, block_size), dtype=torch.long, device=device)
        y = torch.zeros((batch_size, block_size), dtype=torch.long, device=device)
        return x, y

    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

import torch
from torch.utils.data import Dataset



class DLoader_iwslt(Dataset):
    def __init__(self, data, tokenizers, config):
        self.data = data
        self.src_tokenizer, self.trg_tokenizer = tokenizers[0], tokenizers[1]
        self.max_len = config.max_len
        self.ende = config.ende
        self.length = len(self.data)

    
    def add_special_token(self, s, tokenizer):
        s = [tokenizer.sos_token_id] + tokenizer.encode(s)[:self.max_len-2] + [tokenizer.eos_token_id]
        s = s + [tokenizer.pad_token_id] * (self.max_len - len(s))
        return s


    def __getitem__(self, idx):
        if self.ende:
            src, trg = self.add_special_token(self.data[idx][0], self.src_tokenizer), self.add_special_token(self.data[idx][1], self.trg_tokenizer)
        else:
            src, trg = self.add_special_token(self.data[idx][1], self.src_tokenizer), self.add_special_token(self.data[idx][0], self.trg_tokenizer)
        return torch.LongTensor(src), torch.LongTensor(trg)

    
    def __len__(self):
        return self.length



class DLoader_wmt(Dataset):
    def __init__(self, data, tokenizers, config):
        self.data = data
        self.src_tokenizer, self.trg_tokenizer = tokenizers[0], tokenizers[1]
        self.max_len = config.max_len
        self.ende = config.ende
        self.length = len(self.data)

    
    def add_special_token(self, s, tokenizer):
        s = [tokenizer.sos_token_id] + tokenizer.encode(s)[:self.max_len-2] + [tokenizer.eos_token_id]
        s = s + [tokenizer.pad_token_id] * (self.max_len - len(s))
        return s


    def __getitem__(self, idx):
        if self.ende:
            src, trg = self.add_special_token(self.data[idx][0], self.src_tokenizer), self.add_special_token(self.data[idx][1], self.trg_tokenizer)
        else:
            src, trg = self.add_special_token(self.data[idx][1], self.src_tokenizer), self.add_special_token(self.data[idx][0], self.trg_tokenizer)
        return torch.LongTensor(src), torch.LongTensor(trg)

    
    def __len__(self):
        return self.length
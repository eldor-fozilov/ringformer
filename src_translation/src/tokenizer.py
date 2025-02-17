from transformers import BertTokenizer



class Tokenizer_iwslt_ende:
    def __init__(self, config):
        self.tokenizer = BertTokenizer(vocab_file=config.tokenizer_path, do_lower_case=False) 

        self.pad_token, self.pad_token_id = self.tokenizer.pad_token, self.tokenizer.pad_token_id
        self.sos_token, self.sos_token_id = self.tokenizer.cls_token, self.tokenizer.cls_token_id
        self.eos_token, self.eos_token_id = self.tokenizer.sep_token, self.tokenizer.sep_token_id
        self.unk_token, self.unk_token_id = self.tokenizer.unk_token, self.tokenizer.unk_token_id

        self.vocab_size = len(self.tokenizer)


    def tokenize(self, s):
        return self.tokenizer.tokenize(s)


    def encode(self, s):
        # for eliminate sos and eos tokens that are automatically added to a sentence
        return self.tokenizer.encode(s)[1:-1]


    def decode(self, tok):
        try:
            tok = tok[:tok.index(self.eos_token_id)]
        except ValueError:
            try:
                tok = tok[:tok.index(self.pad_token_id)]
            except:
                pass
        return self.tokenizer.decode(tok)




class Tokenizer_wmt_ende:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("jhu-clsp/bibert-ende")

        self.pad_token, self.pad_token_id = self.tokenizer.pad_token, self.tokenizer.pad_token_id
        self.sos_token, self.sos_token_id = self.tokenizer.cls_token, self.tokenizer.cls_token_id
        self.eos_token, self.eos_token_id = self.tokenizer.sep_token, self.tokenizer.sep_token_id
        self.unk_token, self.unk_token_id = self.tokenizer.unk_token, self.tokenizer.unk_token_id

        self.vocab_size = len(self.tokenizer)


    def tokenize(self, s):
        return self.tokenizer.tokenize(s)


    def encode(self, s):
        # for eliminate sos and eos tokens that are automatically added to a sentence
        return self.tokenizer.encode(s)[1:-1]


    def decode(self, tok):
        try:
            tok = tok[:tok.index(self.eos_token_id)]
        except ValueError:
            try:
                tok = tok[:tok.index(self.pad_token_id)]
            except:
                pass
        return self.tokenizer.decode(tok)
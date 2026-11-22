import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

from transformers import AddedToken

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        # TODO
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')

        # EC v4: general-purpose SQL vocabulary
        sql_tokens = [
            "SELECT", "DISTINCT", "FROM", "WHERE",
            "GROUP", "BY", "ORDER", "HAVING", "LIMIT",
            "AND", "OR", "NOT", "IN", "BETWEEN",
            "LIKE", "IS", "NULL", "EXISTS",
            "JOIN", "INNER", "LEFT", "RIGHT",
            "OUTER", "CROSS", "ON", "AS",
            "COUNT", "SUM", "AVG", "MIN", "MAX",
            "UNION", "INTERSECT", "EXCEPT",
            "ALL", "ASC", "DESC"
        ]

        self.tokenizer.add_tokens(["<", "<=", ">", ">=", "!=", "<>", "="])

        self.tokenizer.add_tokens([
            AddedToken(token, single_word=True, normalized=False)
            for token in sql_tokens
            if len(self.tokenizer.tokenize(token)) > 1
        ])
        # beginning of sentence token
        # self.start_token_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")
        # use T5's default decoder start token (PAD ID 0)
        self.start_token_id = self.tokenizer.pad_token_id
        self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        # TODO
        self.raw_nl = load_lines(os.path.join(data_folder, f"{split}.nl"))
        self.processed_nl = self.raw_nl.copy()

        # tokenize
        self.encoder_ids = [tokenizer.encode(text) for text in self.processed_nl]

        if split != "test":
            self.raw_sql = load_lines(os.path.join(data_folder, f"{split}.sql"))
            self.processed_sql = self.raw_sql.copy()

            # replace unsupported SQL operator before tokenization
            # self.processed_sql = [sql.replace("<", "LT_OPERATOR") for sql in self.raw_sql]

            self.sql_ids = [tokenizer.encode(sql) for sql in self.processed_sql]

            assert len(self.encoder_ids) == len(self.sql_ids)
        else: # no golden sql
            self.raw_sql = None
            self.processed_sql = None
            self.sql_ids = None

    
    def __len__(self):
        # TODO
        return len(self.encoder_ids)

    def __getitem__(self, idx):
        # TODO
        encoder_ids = torch.tensor(self.encoder_ids[idx], dtype=torch.long)
        start_id = torch.tensor(self.start_token_id, dtype=torch.long)

        # test split, return only encoder id and start id
        if self.split == "test":
            return encoder_ids, start_id

        # train split, return also the sql ids
        sql_ids = torch.tensor(self.sql_ids[idx], dtype=torch.long)
        return encoder_ids, sql_ids, start_id

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    encoder_seqs, sql_seqs, start_ids = zip(*batch)

    # shape BxT (batch * longest seq len of the batch)
    encoder_ids = pad_sequence(encoder_seqs, batch_first=True, padding_value=PAD_IDX)
    # 1 is real token and 0 is padding
    encoder_mask = (encoder_ids != PAD_IDX).long()

    decoder_targets = pad_sequence(sql_seqs, batch_first=True, padding_value=PAD_IDX)
    # shift right by 1, add beginning sentence token
    decoder_input_seqs = [torch.cat([start_id.unsqueeze(0), sql_ids[:-1]])
                          for sql_ids, start_id in zip(sql_seqs, start_ids)]
    decoder_inputs = pad_sequence(decoder_input_seqs, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.stack(start_ids).unsqueeze(1)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    encoder_seqs, start_ids = zip(*batch)
    
    encoder_ids = pad_sequence(encoder_seqs, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.stack(start_ids).unsqueeze(1)

    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))

    return train_x, train_y, dev_x, dev_y, test_x
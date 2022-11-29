import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, BertConfig
from typing import Callable

class Retriever(nn.Module):
    def __init__(self, config, tokenizer):
        super(Retriever, self).__init__()
        self.config = config

        bert_config = BertConfig.from_pretrained(config.model_type)
        self.encoder = BertModel.from_pretrained(config.model_type, return_dict=True)

        self.hidden_size = bert_config.hidden_size
        self.hidden_dropout_prob = bert_config.hidden_dropout_prob
        self.num_hidden_layers = bert_config.num_hidden_layers
        self.pad_id = bert_config.pad_token_id

        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def forward(self, contents, candidate_idioms, mask_locations):
        """

        :param contents: [bsz, len_content]
        :param candidate_idioms: [bsz, 7, candidate_idiom]
        :param mask_locations: [bsz]
        :return:
            dot_product: [bsz, 7]
        """
        bsz, can_num, idiom_len = candidate_idioms.shape

        # TODO
        # no "pooler_output", use mask location!!
        contents_attention_mask = get_mask(contents, self.pad_id)
        contents_rep = self.encoder(input_ids=contents,
                                   attention_mask=contents_attention_mask).last_hidden_state # [bsz, len_content, dim]

        bsz, len_content, dim = contents_rep.shape
        mask_locations = mask_locations.repeat(dim, 1) # [dim, bsz]
        mask_locations = mask_locations.permute(1,0) # [bsz, dim]
        mask_locations = mask_locations.unsqueeze(1) # [bsz, 1, dim]
        temp_contents = torch.gather(contents_rep, 1, mask_locations) # [bsz, 1, dim],  value in the second dim is selected by mask location
        temp_contents = temp_contents.squeeze(1) # [bsz, dim]
        contents_rep = self.dropout(temp_contents) #

        # encode candidate idiom
        candidate_input_ids_flatten = candidate_idioms.view(-1, idiom_len)
        candidate_attention_mask_flatten = get_mask(candidate_input_ids_flatten, self.pad_id)


        candidate_rep_flatten = self.encoder(input_ids=candidate_input_ids_flatten,
                                             attention_mask=candidate_attention_mask_flatten).pooler_output
        candidate_rep = candidate_rep_flatten.view(-1, can_num, self.hidden_size)  # [bsz, can_num(7), dim]
        candidate_rep = self.dropout(candidate_rep)

        # [bsz, can_num, dim] * [bsz, dim, 1]
        dot_product = torch.matmul(candidate_rep, contents_rep.unsqueeze(-1)).squeeze(-1) / np.sqrt(self.hidden_size)

        eps = -1e8

        return dot_product

def get_mask(input_ids: torch.LongTensor, pad_id: int):
    return (~input_ids.eq(pad_id)).float()
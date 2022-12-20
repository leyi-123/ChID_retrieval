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
        if self.config.use_mask:
            contents_rep = self.encoder(input_ids=contents,
                                       attention_mask=contents_attention_mask).last_hidden_state # [bsz, len_content, dim]

            bsz, len_content, dim = contents_rep.shape
            #mask_locations = mask_locations.repeat(dim, 1) # [dim, bsz]
            #mask_locations = mask_locations.permute(1,0) # [bsz, dim]
            #mask_locations = mask_locations.unsqueeze(1) # [bsz, 1, dim]
            #temp_contents = torch.gather(contents_rep, 1, mask_locations) # [bsz, 1, dim],  value in the second dim is selected by mask location

            temp_contents = contents_rep[0, mask_locations[0], :]
            temp_contents = temp_contents.unsqueeze(0)
            for i in range(1, bsz):
                t = contents_rep[i, mask_locations[i], :]
                t = t.unsqueeze(0)
                temp_contents = torch.cat((temp_contents, t), dim=0)
            #temp_contents = temp_contents.squeeze(1) # [bsz, dim]
            contents_rep = self.dropout(temp_contents) #
        else:
            contents_rep = self.encoder(input_ids=contents,
                                        attention_mask=contents_attention_mask).pooler_output  # [bsz, dim]
            contents_rep = self.dropout(contents_rep)  #

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

class Cross_Retriever(nn.Module):
    def __init__(self, config, tokenizer):
        super(Cross_Retriever, self).__init__()
        self.config = config

        bert_config = BertConfig.from_pretrained(config.model_type)
        self.encoder = BertModel.from_pretrained(config.model_type, return_dict=True)

        self.hidden_size = bert_config.hidden_size
        self.hidden_dropout_prob = bert_config.hidden_dropout_prob
        self.num_hidden_layers = bert_config.num_hidden_layers
        self.pad_id = bert_config.pad_token_id

        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, 1)

    def forward(self, idiom_contents, mask_locations):
        """

        :param idiom_contents: [bsz, 7, idiom_content_len]
        :param mask_locations: [bsz]
        :return:
            dot_product: [bsz, 7]
        """
        bsz, can_num, idiom_content_len = idiom_contents.shape

        # TODO
        # no "pooler_output", use mask location!!

        if self.config.use_mask:
            idiom_contents_input_ids_flatten = idiom_contents.view(-1, idiom_content_len) # [bsz * can_num(7), idiom_content_len]
            idiom_contents_attention_mask_flatten = get_mask(idiom_contents_input_ids_flatten, self.pad_id)

            idiom_contents_rep_flatten = self.encoder(input_ids=idiom_contents_input_ids_flatten,
                                                 attention_mask=idiom_contents_attention_mask_flatten).last_hidden_state # [bsz * can_num(7), len_content, dim]
            idiom_contenes_rep_c = idiom_contents_rep_flatten.view(bsz, can_num, idiom_content_len, -1) # [bsz , can_num(7), len_content, dim]
            #mask_locations = mask_locations.repeat(dim, 1) # [dim, bsz]
            #mask_locations = mask_locations.permute(1,0) # [bsz, dim]
            #mask_locations = mask_locations.unsqueeze(1) # [bsz, 1, dim]
            #temp_contents = torch.gather(contents_rep, 1, mask_locations) # [bsz, 1, dim],  value in the second dim is selected by mask location

            temp_idiom_contents = idiom_contenes_rep_c[0, :, mask_locations[0], :]
            temp_idiom_contents = temp_idiom_contents.unsqueeze(0)
            for i in range(1, bsz):
                t = idiom_contenes_rep_c[i, :, mask_locations[i], :]
                t = t.unsqueeze(0)
                temp_idiom_contents = torch.cat((temp_idiom_contents, t), dim=0)
            #temp_contents = temp_contents.squeeze(1) # [bsz, dim]
            idiom_contents_rep = self.dropout(temp_idiom_contents) # [bsz, can_num, dim]
        else:
            idiom_contents_input_ids_flatten = idiom_contents.view(-1,
                                                                   idiom_content_len)  # [bsz * can_num(7), idiom_content_len]
            idiom_contents_attention_mask_flatten = get_mask(idiom_contents_input_ids_flatten, self.pad_id)

            idiom_contents_rep_flatten = self.encoder(input_ids=idiom_contents_input_ids_flatten,
                                                      attention_mask=idiom_contents_attention_mask_flatten).pooler_output  # [bsz * can_num(7), dim]
            idiom_contenes_rep_c = idiom_contents_rep_flatten.view(bsz, can_num, -1)
            idiom_contents_rep = self.dropout(idiom_contenes_rep_c)  #

        idiom_contents_rep = idiom_contents_rep.view(bsz*can_num, -1)
        logits = self.classifier(idiom_contents_rep).squeeze(-1).view(bsz, can_num)

        return logits
class Cross_Retriever_mask(nn.Module):
    def __init__(self, config, tokenizer):
        super(Cross_Retriever_mask, self).__init__()
        self.config = config

        bert_config = BertConfig.from_pretrained(config.model_type)
        self.encoder = BertModel.from_pretrained(config.model_type, return_dict=True)

        self.hidden_size = bert_config.hidden_size
        self.hidden_dropout_prob = bert_config.hidden_dropout_prob
        self.num_hidden_layers = bert_config.num_hidden_layers
        self.pad_id = bert_config.pad_token_id

        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, 1)

    def forward(self, idiom_contents):
        """

        :param idiom_contents: [bsz, 7, idiom_content_len]
        :return:
            dot_product: [bsz, 7]
        """
        bsz, can_num, idiom_content_len = idiom_contents.shape

        idiom_contents_input_ids_flatten = idiom_contents.view(-1,
                                                               idiom_content_len)  # [bsz * can_num(7), idiom_content_len]
        idiom_contents_attention_mask_flatten = get_mask(idiom_contents_input_ids_flatten, self.pad_id)

        idiom_contents_rep_flatten = self.encoder(input_ids=idiom_contents_input_ids_flatten,
                                                  attention_mask=idiom_contents_attention_mask_flatten).pooler_output  # [bsz * can_num(7), dim]
        idiom_contenes_rep_c = idiom_contents_rep_flatten.view(bsz, can_num, -1)
        idiom_contents_rep = self.dropout(idiom_contenes_rep_c)  #

        idiom_contents_rep = idiom_contents_rep.view(bsz*can_num, -1)
        logits = self.classifier(idiom_contents_rep).squeeze(-1).view(bsz, can_num)

        return logits

def get_mask(input_ids: torch.LongTensor, pad_id: int):
    return (~input_ids.eq(pad_id)).float()
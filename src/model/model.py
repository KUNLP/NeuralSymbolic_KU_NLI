# model += Parsing Infor Collecting Layer (PIC)

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn as nn
import torch
import torch.nn.functional as F

#from transformers.modeling_electra import ElectraModel, ElectraPreTrainedModel

from transformers import ElectraModel, RobertaModel

import transformers
if int(transformers.__version__[0]) <= 3:
    from transformers.modeling_roberta import RobertaPreTrainedModel
    from transformers.modeling_bert import BertPreTrainedModel
    from transformers.modeling_electra import ElectraModel, ElectraPreTrainedModel
else:
    from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
    from transformers.models.bert.modeling_bert import BertPreTrainedModel
    from transformers.models.electra.modeling_electra import ElectraPreTrainedModel

from src.functions.biattention import BiAttention, BiLinear

class RobertaForSequenceClassification(BertPreTrainedModel):

    def __init__(self, config, prem_max_sentence_length, hypo_max_sentence_length):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = RobertaModel(config)

        # 입력 토큰에서 token1, token2가 있을 때 (index of token1, index of token2)를 하나의 span으로 보고 이에 대한 정보를 학습
        self.span_info_collect = SICModel1(config)
        #self.span_info_collect = SICModel2(config)

        # biaffine을 통해 premise와 hypothesis span에 대한 정보를 결합후 정규화
        self.parsing_info_collect = PICModel1(config, prem_max_sentence_length, hypo_max_sentence_length) # 구묶음 + tag 정보 + klue-biaffine attention + bilistm + klue-bilinear classification
        #self.parsing_info_collect = PICModel2(config, prem_max_sentence_length, hypo_max_sentence_length) # 구묶음 + bilistm + klue-bilinear classification
        #self.parsing_info_collect = PICModel3(config, prem_max_sentence_length, hypo_max_sentence_length) # 구묶음 + tag 정보 + bilistm + klue-bilinear classification
        #self.parsing_info_collect = PICModel4(config, prem_max_sentence_length, hypo_max_sentence_length)  # 구묶음 + tag 정보(1) + bilistm + bilinear classification
        #self.parsing_info_collect = PICModel5(config, prem_max_sentence_length, hypo_max_sentence_length)  # 구묶음 + tag 정보 + biaffine attention + bilistm + bilinear classification

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        prem_span=None,
        hypo_span=None,
        prem_word_idxs=None,
        hypo_word_idxs=None,
    ):
        batch_size = input_ids.shape[0]
        discriminator_hidden_states = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # last-layer hidden state
        # sequence_output: [batch_size, seq_length, hidden_size]
        sequence_output = discriminator_hidden_states[0]

        # span info collecting layer(SIC)
        h_ij = self.span_info_collect(sequence_output, prem_word_idxs, hypo_word_idxs)

        # parser info collecting layer(PIC)
        logits = self.parsing_info_collect(h_ij,
                                           batch_size= batch_size,
                                      prem_span=prem_span,hypo_span=hypo_span,)

        outputs = (logits, ) + discriminator_hidden_states[2:]

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            print("loss: "+str(loss))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)



class SICModel1(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, hidden_states, prem_word_idxs, hypo_word_idxs):
        # (batch, max_pre_sen, seq_len) @ (batch, seq_len, hidden) = (batch, max_pre_sen, hidden)
        prem_word_idxs = prem_word_idxs.squeeze(1)
        hypo_word_idxs = hypo_word_idxs.squeeze(1)

        prem = torch.matmul(prem_word_idxs, hidden_states)
        hypo = torch.matmul(hypo_word_idxs, hidden_states)

        return [prem, hypo]

class SICModel2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.W_p_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_p_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_p_3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_p_4 = nn.Linear(self.hidden_size, self.hidden_size)

        self.W_h_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_h_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_h_3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_h_4 = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, hidden_states, prem_word_idxs, hypo_word_idxs):
        prem_word_idxs = prem_word_idxs.squeeze(1).type(torch.LongTensor).to("cuda")
        hypo_word_idxs = hypo_word_idxs.squeeze(1).type(torch.LongTensor).to("cuda")

        Wp1_h = self.W_p_1(hidden_states)  # (bs, length, hidden_size)
        Wp2_h = self.W_p_2(hidden_states)
        Wp3_h = self.W_p_3(hidden_states)
        Wp4_h = self.W_p_4(hidden_states)

        Wh1_h = self.W_h_1(hidden_states)  # (bs, length, hidden_size)
        Wh2_h = self.W_h_2(hidden_states)
        Wh3_h = self.W_h_3(hidden_states)
        Wh4_h = self.W_h_4(hidden_states)

        W1_hi_emb=torch.tensor([], dtype=torch.long).to("cuda")
        W2_hi_emb=torch.tensor([], dtype=torch.long).to("cuda")
        W3_hi_start_emb = torch.tensor([], dtype=torch.long).to("cuda")
        W3_hi_end_emb = torch.tensor([], dtype=torch.long).to("cuda")
        W4_hi_start_emb = torch.tensor([], dtype=torch.long).to("cuda")
        W4_hi_end_emb = torch.tensor([], dtype=torch.long).to("cuda")
        for i in range(0, hidden_states.shape[0]):
            sub_W1_hi_emb = torch.index_select(Wp1_h[i], 0, prem_word_idxs[i][0])  # (prem_max_seq_length, hidden_size)
            sub_W2_hi_emb = torch.index_select(Wp2_h[i], 0, prem_word_idxs[i][1])
            sub_W3_hi_start_emb = torch.index_select(Wp3_h[i], 0, prem_word_idxs[i][0])
            sub_W3_hi_end_emb = torch.index_select(Wp3_h[i], 0, prem_word_idxs[i][1])
            sub_W4_hi_start_emb = torch.index_select(Wp4_h[i], 0, prem_word_idxs[i][0])
            sub_W4_hi_end_emb = torch.index_select(Wp4_h[i], 0, prem_word_idxs[i][1])

            W1_hi_emb = torch.cat((W1_hi_emb, sub_W1_hi_emb.unsqueeze(0)))
            W2_hi_emb = torch.cat((W2_hi_emb, sub_W2_hi_emb.unsqueeze(0)))
            W3_hi_start_emb = torch.cat((W3_hi_start_emb, sub_W3_hi_start_emb.unsqueeze(0)))
            W3_hi_end_emb = torch.cat((W3_hi_end_emb, sub_W3_hi_end_emb.unsqueeze(0)))
            W4_hi_start_emb = torch.cat((W4_hi_start_emb, sub_W4_hi_start_emb.unsqueeze(0)))
            W4_hi_end_emb = torch.cat((W4_hi_end_emb, sub_W4_hi_end_emb.unsqueeze(0)))

        # [w1*hi, w2*hj, w3(hi-hj), w4(hi⊗hj)]
        prem_span = W1_hi_emb + W2_hi_emb + (W3_hi_start_emb - W3_hi_end_emb) + torch.mul(W4_hi_start_emb, W4_hi_end_emb) # (batch_size, prem_max_seq_length, hidden_size)
        prem_h_ij = torch.tanh(prem_span)

        W1_hi_emb = torch.tensor([], dtype=torch.long).to("cuda")
        W2_hi_emb = torch.tensor([], dtype=torch.long).to("cuda")
        W3_hi_start_emb = torch.tensor([], dtype=torch.long).to("cuda")
        W3_hi_end_emb = torch.tensor([], dtype=torch.long).to("cuda")
        W4_hi_start_emb = torch.tensor([], dtype=torch.long).to("cuda")
        W4_hi_end_emb = torch.tensor([], dtype=torch.long).to("cuda")
        for i in range(0, hidden_states.shape[0]):
            sub_W1_hi_emb = torch.index_select(Wh1_h[i], 0, hypo_word_idxs[i][0])  # (hypo_max_seq_length, hidden_size)
            sub_W2_hi_emb = torch.index_select(Wh2_h[i], 0, hypo_word_idxs[i][1])
            sub_W3_hi_start_emb = torch.index_select(Wh3_h[i], 0, hypo_word_idxs[i][0])
            sub_W3_hi_end_emb = torch.index_select(Wh3_h[i], 0, hypo_word_idxs[i][1])
            sub_W4_hi_start_emb = torch.index_select(Wh4_h[i], 0, hypo_word_idxs[i][0])
            sub_W4_hi_end_emb = torch.index_select(Wh4_h[i], 0, hypo_word_idxs[i][1])

            W1_hi_emb = torch.cat((W1_hi_emb, sub_W1_hi_emb.unsqueeze(0)))
            W2_hi_emb = torch.cat((W2_hi_emb, sub_W2_hi_emb.unsqueeze(0)))
            W3_hi_start_emb = torch.cat((W3_hi_start_emb, sub_W3_hi_start_emb.unsqueeze(0)))
            W3_hi_end_emb = torch.cat((W3_hi_end_emb, sub_W3_hi_end_emb.unsqueeze(0)))
            W4_hi_start_emb = torch.cat((W4_hi_start_emb, sub_W4_hi_start_emb.unsqueeze(0)))
            W4_hi_end_emb = torch.cat((W4_hi_end_emb, sub_W4_hi_end_emb.unsqueeze(0)))

        # [w1*hi, w2*hj, w3(hi-hj), w4(hi⊗hj)]
        hypo_span = W1_hi_emb + W2_hi_emb + (W3_hi_start_emb - W3_hi_end_emb) + torch.mul(W4_hi_start_emb, W4_hi_end_emb)  # (batch_size, hypo_max_seq_length, hidden_size)
        hypo_h_ij = torch.tanh(hypo_span)

        h_ij = [prem_h_ij, hypo_h_ij]

        return h_ij


class PICModel1(nn.Module):
    def __init__(self, config, prem_max_sentence_length, hypo_max_sentence_length):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.prem_max_sentence_length = prem_max_sentence_length
        self.hypo_max_sentence_length = hypo_max_sentence_length
        self.num_labels = config.num_labels

        # 구문구조 종류
        depend2idx = {"None": 0};
        idx2depend = {0: "None"};
        for depend1 in ['IP', 'AP', 'DP', 'VP', 'VNP', 'S', 'R', 'NP', 'L', 'X']:
            for depend2 in ['CMP', 'MOD', 'SBJ', 'AJT', 'CNJ', 'None', 'OBJ', "UNDEF"]:
                depend2idx[depend1 + "-" + depend2] = len(depend2idx)
                idx2depend[len(idx2depend)] = depend1 + "-" + depend2
        self.depend2idx = depend2idx
        self.idx2depend = idx2depend
        self.depend_embedding = nn.Embedding(len(idx2depend), self.hidden_size, padding_idx=0).to("cuda")

        self.reduction1 = nn.Linear(self.hidden_size , int(self.hidden_size // 3))
        self.reduction2 = nn.Linear(self.hidden_size , int(self.hidden_size // 3))
        self.reduction3 = nn.Linear(self.hidden_size, int(self.hidden_size // 3))
        self.reduction4 = nn.Linear(self.hidden_size, int(self.hidden_size // 3))

        self.biaffine1 = BiAttention(int(self.hidden_size // 3), int(self.hidden_size // 3), 100)
        self.biaffine2 = BiAttention(int(self.hidden_size // 3), int(self.hidden_size // 3), 100)

        self.bi_lism_1 = nn.LSTM(input_size=100, hidden_size=self.hidden_size//2, num_layers=1, bidirectional=True)
        self.bi_lism_2 = nn.LSTM(input_size=100, hidden_size=self.hidden_size//2, num_layers=1, bidirectional=True)

        self.bilinear = BiLinear(self.hidden_size, self.hidden_size, self.num_labels)

        # self.W_1_bilinear = nn.Bilinear(int(self.hidden_size // 3), int(self.hidden_size // 3), self.hidden_size, bias=False)
        # self.W_1_linear = nn.Linear(int(self.hidden_size // 3) + int(self.hidden_size // 3), self.hidden_size)
        # self.W_2_bilinear = nn.Bilinear(int(self.hidden_size // 3), int(self.hidden_size // 3), self.hidden_size, bias=False)
        # self.W_2_linear = nn.Linear(int(self.hidden_size // 3) + int(self.hidden_size // 3), self.hidden_size)
        #
        # self.bi_lism_1 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.prem_max_sentence_length//2, num_layers=1, bidirectional=True)
        # self.bi_lism_2 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hypo_max_sentence_length//2, num_layers=1, bidirectional=True)
        #
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 일반화된 정보를 사용
        # self.biaffine_W_bilinear = nn.Linear((2*(self.prem_max_sentence_length//2))*(2*(self.hypo_max_sentence_length//2)), self.num_labels, bias=False)
        # self.biaffine_W_linear = nn.Linear(2*(self.prem_max_sentence_length//2) + 2*(self.hypo_max_sentence_length//2), self.num_labels)
        # #self.biaffine_W_bilinear = nn.Bilinear(self.hidden_size, self.hidden_size, self.num_labels, bias=False)
        # #self.biaffine_W_linear = nn.Linear(self.hidden_size*2, self.num_labels)
        #
        # self.reset_parameters()

    def forward(self, hidden_states, batch_size, prem_span, hypo_span):
        # hidden_states: [[batch_size, word_idxs, hidden_size], []]
        # span: [batch_size, max_sentence_length, max_sentence_length]
        # word_idxs: [batch_size, seq_length]
        # -> sequence_outputs: [batch_size, seq_length, hidden_size]

        prem_hidden_states= hidden_states[0]
        hypo_hidden_states= hidden_states[1]
        #print(prem_hidden_states.shape, hypo_hidden_states.shape, prem_span.shape, hypo_span.shape)

        # span: (batch, max_prem_len, 3) -> (batch, max_prem_len, 3*hidden_size)
        new_prem_span = torch.tensor([], dtype=torch.long).to("cuda")
        new_hypo_span = torch.tensor([], dtype=torch.long).to("cuda")

        for i, (p_span, h_span) in enumerate(zip(prem_span.tolist(), hypo_span.tolist())):
            p_span_head = torch.tensor([span[0] for span in p_span]).to("cuda") #(max_prem_len)
            p_span_tail = torch.tensor([span[1] for span in p_span]).to("cuda")
            p_span_dep = torch.tensor([span[2] for span in p_span]).to("cuda")

            p_span_head = torch.index_select(prem_hidden_states[i], 0, p_span_head) #(max_prem_len, hidden_size)
            p_span_tail = torch.index_select(prem_hidden_states[i], 0, p_span_tail)
            p_span_dep = self.depend_embedding(p_span_dep)

            n_p_span = p_span_head + p_span_tail + p_span_dep
            new_prem_span = torch.cat((new_prem_span, n_p_span.unsqueeze(0)))

            h_span_head = torch.tensor([span[0] for span in h_span]).to("cuda")  # (max_hypo_len)
            h_span_tail = torch.tensor([span[1] for span in h_span]).to("cuda")
            h_span_dep = torch.tensor([span[2] for span in h_span]).to("cuda")

            h_span_head = torch.index_select(hypo_hidden_states[i], 0, h_span_head)  # (max_hypo_len, hidden_size)
            h_span_tail = torch.index_select(hypo_hidden_states[i], 0, h_span_tail)
            h_span_dep = self.depend_embedding(h_span_dep)

            n_h_span = h_span_head + h_span_tail + h_span_dep
            new_hypo_span = torch.cat((new_hypo_span, n_h_span.unsqueeze(0)))

        prem_span = new_prem_span
        hypo_span = new_hypo_span

        del new_prem_span
        del new_hypo_span

        # biaffine attention
        # hidden_states: (batch_size, max_prem_len, hidden_size)
        # span: (batch, max_prem_len, hidden_size)
        # -> biaffine_outputs: [batch_size, 100, max_prem_len,  max_prem_len]
        prem_span = self.reduction1(prem_span)
        prem_hidden_states = self.reduction2(prem_hidden_states)
        hypo_span = self.reduction3(hypo_span)
        hypo_hidden_states = self.reduction4(hypo_hidden_states)

        prem_biaffine_outputs= self.biaffine1(prem_hidden_states, prem_span)
        hypo_biaffine_outputs = self.biaffine2(hypo_hidden_states, hypo_span)

        # outputs = self.bilinear(prem_biaffine_outputs.view(-1,self.prem_max_sentence_length*self.prem_max_sentence_length),
        #                         hypo_biaffine_outputs.view(-1,self.hypo_max_sentence_length*self.hypo_max_sentence_length))

        # bilstm
        # biaffine_outputs: [batch_size, 100, max_prem_len,  max_prem_len] -> [batch_size, 100, max_prem_len] -> [max_prem_len, batch_size, 100]
        # -> hidden_states: [batch_size, max_sentence_length]
        prem_biaffine_outputs = prem_biaffine_outputs.mean(-1)
        hypo_biaffine_outputs = hypo_biaffine_outputs.mean(-1)

        prem_biaffine_outputs = prem_biaffine_outputs.transpose(1,2).transpose(0,1)
        hypo_biaffine_outputs = hypo_biaffine_outputs.transpose(1,2).transpose(0,1)

        prem_states = None
        hypo_states = None

        prem_bilstm_outputs, prem_states = self.bi_lism_1(prem_biaffine_outputs)
        hypo_bilstm_outputs, hypo_states = self.bi_lism_2(hypo_biaffine_outputs)


        prem_hidden_states = prem_states[0].transpose(0, 1).contiguous().view(batch_size, -1)
        hypo_hidden_states = hypo_states[0].transpose(0, 1).contiguous().view(batch_size, -1)

        outputs = self.bilinear(prem_hidden_states, hypo_hidden_states)

        # new_prem_hidden_states = prem_hidden_states.view(-1, self.prem_max_sentence_length, 1, self.hidden_size) # (batch_size, max_prem_len, 1, hidden_size)
        # new_hypo_hidden_states = hypo_hidden_states.view(-1, self.hypo_max_sentence_length, 1, self.hidden_size)
        # new_prem_span = prem_span.view(-1, self.prem_max_sentence_length, 3, self.hidden_size)# (batch_size, max_prem_len, 3, hidden_size)
        # new_hypo_span = hypo_span.view(-1, self.hypo_max_sentence_length, 3, self.hidden_size)
        #
        # prem_depend = (new_prem_hidden_states.unsqueeze(-1) * new_prem_span.unsqueeze(-2)).view(-1, self.prem_max_sentence_length, 3*self.hidden_size, self.hidden_size) # (batch_size, max_prem_len, 3*hidden_size, hidden_size)
        # prem_depend = self.reduction1(prem_depend.transpose(2,3)).view(-1, self.prem_max_sentence_length, int(3 * self.hidden_size // 64)*self.hidden_size) # (batch_size, max_prem_len, hidden_size, 3*hidden_size) -> (batch_size, max_prem_len, hidden_size, int(3 * self.hidden_size // 64))
        #
        # hypo_depend = (new_hypo_hidden_states.unsqueeze(-1) * new_hypo_span.unsqueeze(-2)).view(-1, self.hypo_max_sentence_length, 3*self.hidden_size, self.hidden_size)
        # hypo_depend = self.reduction2(hypo_depend.transpose(2,3)).view(-1, self.hypo_max_sentence_length, int(3 * self.hidden_size // 64)*self.hidden_size)
        #
        # prem_biaffine_outputs= self.W_1_bilinear(prem_depend) + self.W_1_linear(torch.cat((prem_span, prem_hidden_states), dim = -1))
        # hypo_biaffine_outputs = self.W_2_bilinear(hypo_depend) + self.W_2_linear(torch.cat((hypo_span, hypo_hidden_states), dim = -1))
        #
        # # bilstm
        # # biaffine_outputs: [batch_size, max_sentence_length, hidden_size]
        # # -> hidden_states: [batch_size, max_sentence_length]
        # prem_biaffine_outputs = prem_biaffine_outputs.transpose(0,1)
        # hypo_biaffine_outputs = hypo_biaffine_outputs.transpose(0,1)
        #
        # prem_states = None
        # hypo_states = None
        #
        # prem_bilstm_outputs, prem_states = self.bi_lism_1(prem_biaffine_outputs)
        # hypo_bilstm_outputs, hypo_states = self.bi_lism_2(hypo_biaffine_outputs)
        #
        # prem_hidden_states = prem_states[0].transpose(0, 1).contiguous().view(batch_size, -1)
        # hypo_hidden_states = hypo_states[0].transpose(0, 1).contiguous().view(batch_size, -1)
        # # biaffine attention
        # # prem_hidden_states: (batch_size, max_prem_len)
        # # hypo_hidden_states: (batch_size, max_hypo_len)
        #
        # prem_hypo = (prem_hidden_states.unsqueeze(-1) * hypo_hidden_states.unsqueeze(-2)).view(-1,
        #                                                                                 (2*(self.prem_max_sentence_length//2))*(2*(self.hypo_max_sentence_length//2)))
        #
        # outputs = self.biaffine_W_bilinear(prem_hypo) + self.biaffine_W_linear(torch.cat((prem_hidden_states, hypo_hidden_states), dim=-1))

        return outputs

    def reset_parameters(self):
        self.W_1_bilinear.reset_parameters()
        self.W_1_linear.reset_parameters()
        self.W_2_bilinear.reset_parameters()
        self.W_2_linear.reset_parameters()

        self.biaffine_W_bilinear.reset_parameters()
        self.biaffine_W_linear.reset_parameters()



class PICModel2(nn.Module):
    def __init__(self, config, prem_max_sentence_length, hypo_max_sentence_length):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.prem_max_sentence_length = prem_max_sentence_length
        self.hypo_max_sentence_length = hypo_max_sentence_length
        self.num_labels = config.num_labels

        self.reduction1 = nn.Linear(self.hidden_size , int(self.hidden_size // 3))
        self.reduction2 = nn.Linear(self.hidden_size , int(self.hidden_size // 3))

        self.bi_lism_1 = nn.LSTM(input_size=int(self.hidden_size // 3), hidden_size=self.hidden_size//2, num_layers=1, bidirectional=True)
        self.bi_lism_2 = nn.LSTM(input_size=int(self.hidden_size // 3), hidden_size=self.hidden_size//2, num_layers=1, bidirectional=True)

        self.bilinear = BiLinear(self.hidden_size, self.hidden_size, self.num_labels)

    def forward(self, hidden_states, batch_size, prem_span, hypo_span):
        # hidden_states: [[batch_size, word_idxs, hidden_size], []]
        # span: [batch_size, max_sentence_length, max_sentence_length]
        # word_idxs: [batch_size, seq_length]
        # -> sequence_outputs: [batch_size, seq_length, hidden_size]

        prem_hidden_states= hidden_states[0]
        hypo_hidden_states= hidden_states[1]

        # biLSTM
        # hidden_states: (batch_size, max_prem_len, hidden_size)
        # -> # -> hidden_states: [batch_size, hidden_size]
        prem_hidden_states = self.reduction1(prem_hidden_states)
        hypo_hidden_states = self.reduction2(hypo_hidden_states)
        prem_hidden_states = prem_hidden_states.transpose(0,1)
        hypo_hidden_states = hypo_hidden_states.transpose(0,1)

        prem_bilstm_outputs, prem_states = self.bi_lism_1(prem_hidden_states)
        hypo_bilstm_outputs, hypo_states = self.bi_lism_2(hypo_hidden_states)

        prem_hidden_states = prem_states[0].transpose(0, 1).contiguous().view(batch_size, -1)
        hypo_hidden_states = hypo_states[0].transpose(0, 1).contiguous().view(batch_size, -1)

        # bilinear classification
        outputs = self.bilinear(prem_hidden_states, hypo_hidden_states)

        return outputs


class PICModel3(nn.Module):
    def __init__(self, config, prem_max_sentence_length, hypo_max_sentence_length):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.prem_max_sentence_length = prem_max_sentence_length
        self.hypo_max_sentence_length = hypo_max_sentence_length
        self.num_labels = config.num_labels

        # 구문구조 종류
        depend2idx = {"None": 0};
        idx2depend = {0: "None"};
        for depend1 in ['IP', 'AP', 'DP', 'VP', 'VNP', 'S', 'R', 'NP', 'L', 'X']:
            for depend2 in ['CMP', 'MOD', 'SBJ', 'AJT', 'CNJ', 'None', 'OBJ', "UNDEF"]:
                depend2idx[depend1 + "-" + depend2] = len(depend2idx)
                idx2depend[len(idx2depend)] = depend1 + "-" + depend2
        self.depend2idx = depend2idx
        self.idx2depend = idx2depend
        self.depend_embedding = nn.Embedding(len(idx2depend), self.hidden_size, padding_idx=0).to("cuda")

        self.reduction1 = nn.Linear(self.hidden_size , int(self.hidden_size // 3))
        self.reduction2 = nn.Linear(self.hidden_size , int(self.hidden_size // 3))
        self.reduction3 = nn.Linear(self.hidden_size, int(self.hidden_size // 3))
        self.reduction4 = nn.Linear(self.hidden_size, int(self.hidden_size // 3))

        self.tag1 = BiLinear(int(self.hidden_size // 3), int(self.hidden_size // 3), 100)
        self.tag2 = BiLinear(int(self.hidden_size // 3), int(self.hidden_size // 3), 100)

        self.bi_lism_1 = nn.LSTM(input_size=100, hidden_size=self.hidden_size//2, num_layers=1, bidirectional=True)
        self.bi_lism_2 = nn.LSTM(input_size=100, hidden_size=self.hidden_size//2, num_layers=1, bidirectional=True)

        self.bilinear = BiLinear(self.hidden_size, self.hidden_size, self.num_labels)

    def forward(self, hidden_states, batch_size, prem_span, hypo_span):
        # hidden_states: [[batch_size, word_idxs, hidden_size], []]
        # span: [batch_size, max_sentence_length, max_sentence_length]
        # word_idxs: [batch_size, seq_length]
        # -> sequence_outputs: [batch_size, seq_length, hidden_size]

        prem_hidden_states= hidden_states[0]
        hypo_hidden_states= hidden_states[1]
        #print(prem_hidden_states.shape, hypo_hidden_states.shape, prem_span.shape, hypo_span.shape)

        # span: (batch, max_prem_len, 3) -> (batch, max_prem_len, 3*hidden_size)
        new_prem_span = torch.tensor([], dtype=torch.long).to("cuda")
        new_hypo_span = torch.tensor([], dtype=torch.long).to("cuda")

        for i, (p_span, h_span) in enumerate(zip(prem_span.tolist(), hypo_span.tolist())):
            p_span_head = torch.tensor([span[0] for span in p_span]).to("cuda") #(max_prem_len)
            p_span_tail = torch.tensor([span[1] for span in p_span]).to("cuda")
            p_span_dep = torch.tensor([span[2] for span in p_span]).to("cuda")

            p_span_head = torch.index_select(prem_hidden_states[i], 0, p_span_head) #(max_prem_len, hidden_size)
            p_span_tail = torch.index_select(prem_hidden_states[i], 0, p_span_tail)
            p_span_dep = self.depend_embedding(p_span_dep)

            n_p_span = p_span_head + p_span_tail + p_span_dep
            new_prem_span = torch.cat((new_prem_span, n_p_span.unsqueeze(0)))

            h_span_head = torch.tensor([span[0] for span in h_span]).to("cuda")  # (max_hypo_len)
            h_span_tail = torch.tensor([span[1] for span in h_span]).to("cuda")
            h_span_dep = torch.tensor([span[2] for span in h_span]).to("cuda")

            h_span_head = torch.index_select(hypo_hidden_states[i], 0, h_span_head)  # (max_hypo_len, hidden_size)
            h_span_tail = torch.index_select(hypo_hidden_states[i], 0, h_span_tail)
            h_span_dep = self.depend_embedding(h_span_dep)

            n_h_span = h_span_head + h_span_tail + h_span_dep
            new_hypo_span = torch.cat((new_hypo_span, n_h_span.unsqueeze(0)))

        prem_span = new_prem_span
        hypo_span = new_hypo_span

        del new_prem_span
        del new_hypo_span

        # bilinear
        # hidden_states: (batch_size, max_prem_len, hidden_size)
        # span: (batch, max_prem_len, hidden_size)
        # -> bilinear_outputs: [batch_size, max_prem_len, 100]
        prem_span = self.reduction1(prem_span)
        prem_hidden_states = self.reduction2(prem_hidden_states)
        hypo_span = self.reduction3(hypo_span)
        hypo_hidden_states = self.reduction4(hypo_hidden_states)

        prem_bilinear_outputs= self.tag1(prem_hidden_states, prem_span)
        hypo_bilinear_outputs = self.tag2(hypo_hidden_states, hypo_span)

        # bilstm
        # biaffine_outputs: [batch_size, max_prem_len, 100]
        # -> hidden_states: [batch_size, hidden_size]

        prem_bilinear_outputs = prem_bilinear_outputs. transpose(0,1)
        hypo_bilinear_outputs = hypo_bilinear_outputs.transpose(0,1)

        prem_bilstm_outputs, prem_states = self.bi_lism_1(prem_bilinear_outputs)
        hypo_bilstm_outputs, hypo_states = self.bi_lism_2(hypo_bilinear_outputs)


        prem_hidden_states = prem_states[0].transpose(0, 1).contiguous().view(batch_size, -1)
        hypo_hidden_states = hypo_states[0].transpose(0, 1).contiguous().view(batch_size, -1)

        outputs = self.bilinear(prem_hidden_states, hypo_hidden_states)

        return outputs

class PICModel4(nn.Module):
    def __init__(self, config, prem_max_sentence_length, hypo_max_sentence_length):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.prem_max_sentence_length = prem_max_sentence_length
        self.hypo_max_sentence_length = hypo_max_sentence_length
        self.num_labels = config.num_labels

        self.reduction1 = nn.Linear(self.hidden_size , int(self.hidden_size // 3))
        self.reduction2 = nn.Linear(self.hidden_size , int(self.hidden_size // 3))
        self.reduction3 = nn.Linear(self.hidden_size, int(self.hidden_size // 3))
        self.reduction4 = nn.Linear(self.hidden_size, int(self.hidden_size // 3))

        self.tag1 = BiLinear(int(self.hidden_size // 3), int(self.hidden_size // 3), 100)
        self.tag2 = BiLinear(int(self.hidden_size // 3), int(self.hidden_size // 3), 100)

        self.bi_lism_1 = nn.LSTM(input_size=100, hidden_size=self.hidden_size//2, num_layers=1, bidirectional=True)
        self.bi_lism_2 = nn.LSTM(input_size=100, hidden_size=self.hidden_size//2, num_layers=1, bidirectional=True)

        self.bilinear = BiLinear(self.hidden_size, self.hidden_size, self.num_labels)

    def forward(self, hidden_states, batch_size, prem_span, hypo_span):
        # hidden_states: [[batch_size, word_idxs, hidden_size], []]
        # span: [batch_size, max_sentence_length, max_sentence_length]
        # word_idxs: [batch_size, seq_length]
        # -> sequence_outputs: [batch_size, seq_length, hidden_size]

        prem_hidden_states= hidden_states[0]
        hypo_hidden_states= hidden_states[1]
        #print(prem_hidden_states.shape, hypo_hidden_states.shape, prem_span.shape, hypo_span.shape)

        # span: (batch, max_prem_len, 3) -> (batch, max_prem_len, 3*hidden_size)
        new_prem_span = torch.tensor([], dtype=torch.long).to("cuda")
        new_hypo_span = torch.tensor([], dtype=torch.long).to("cuda")

        for i, (p_span, h_span) in enumerate(zip(prem_span.tolist(), hypo_span.tolist())):
            p_span_head = torch.tensor([span[0] for span in p_span]).to("cuda") #(max_prem_len)
            p_span_tail = torch.tensor([span[1] for span in p_span]).to("cuda")

            p_span_head = torch.index_select(prem_hidden_states[i], 0, p_span_head) #(max_prem_len, hidden_size)
            p_span_tail = torch.index_select(prem_hidden_states[i], 0, p_span_tail)

            n_p_span = p_span_head + p_span_tail
            new_prem_span = torch.cat((new_prem_span, n_p_span.unsqueeze(0)))

            h_span_head = torch.tensor([span[0] for span in h_span]).to("cuda")  # (max_hypo_len)
            h_span_tail = torch.tensor([span[1] for span in h_span]).to("cuda")

            h_span_head = torch.index_select(hypo_hidden_states[i], 0, h_span_head)  # (max_hypo_len, hidden_size)
            h_span_tail = torch.index_select(hypo_hidden_states[i], 0, h_span_tail)

            n_h_span = h_span_head + h_span_tail
            new_hypo_span = torch.cat((new_hypo_span, n_h_span.unsqueeze(0)))

        prem_span = new_prem_span
        hypo_span = new_hypo_span

        del new_prem_span
        del new_hypo_span

        # bilinear
        # hidden_states: (batch_size, max_prem_len, hidden_size)
        # span: (batch, max_prem_len, hidden_size)
        # -> bilinear_outputs: [batch_size, max_prem_len, 100]
        prem_span = self.reduction1(prem_span)
        prem_hidden_states = self.reduction2(prem_hidden_states)
        hypo_span = self.reduction3(hypo_span)
        hypo_hidden_states = self.reduction4(hypo_hidden_states)

        prem_bilinear_outputs= self.tag1(prem_hidden_states, prem_span)
        hypo_bilinear_outputs = self.tag2(hypo_hidden_states, hypo_span)

        # bilstm
        # biaffine_outputs: [batch_size, max_prem_len, 100]
        # -> hidden_states: [batch_size, hidden_size]

        prem_bilinear_outputs = prem_bilinear_outputs. transpose(0,1)
        hypo_bilinear_outputs = hypo_bilinear_outputs.transpose(0,1)

        prem_bilstm_outputs, prem_states = self.bi_lism_1(prem_bilinear_outputs)
        hypo_bilstm_outputs, hypo_states = self.bi_lism_2(hypo_bilinear_outputs)


        prem_hidden_states = prem_states[0].transpose(0, 1).contiguous().view(batch_size, -1)
        hypo_hidden_states = hypo_states[0].transpose(0, 1).contiguous().view(batch_size, -1)

        outputs = self.bilinear(prem_hidden_states, hypo_hidden_states)

        return outputs


class PICModel5(nn.Module):
    def __init__(self, config, prem_max_sentence_length, hypo_max_sentence_length):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.prem_max_sentence_length = prem_max_sentence_length
        self.hypo_max_sentence_length = hypo_max_sentence_length
        self.num_labels = config.num_labels

        # 구문구조 종류
        depend2idx = {"None": 0};
        idx2depend = {0: "None"};
        for depend1 in ['IP', 'AP', 'DP', 'VP', 'VNP', 'S', 'R', 'NP', 'L', 'X']:
            for depend2 in ['CMP', 'MOD', 'SBJ', 'AJT', 'CNJ', 'None', 'OBJ', "UNDEF"]:
                depend2idx[depend1 + "-" + depend2] = len(depend2idx)
                idx2depend[len(idx2depend)] = depend1 + "-" + depend2
        self.depend2idx = depend2idx
        self.idx2depend = idx2depend
        self.depend_embedding = nn.Embedding(len(idx2depend), self.hidden_size, padding_idx=0).to("cuda")

        self.reduction1 = nn.Linear(self.hidden_size , int(self.hidden_size // 6))
        self.reduction2 = nn.Linear(self.hidden_size , int(self.hidden_size // 6))
        self.reduction3 = nn.Linear(self.hidden_size, int(self.hidden_size // 6))
        self.reduction4 = nn.Linear(self.hidden_size, int(self.hidden_size // 6))

        self.W_1_bilinear = nn.Bilinear(int(self.hidden_size // 6), int(self.hidden_size // 6), 100, bias=False)
        self.W_1_linear1 = nn.Linear(int(self.hidden_size // 6), 100)
        self.W_1_linear2 = nn.Linear(int(self.hidden_size // 6), 100)
        self.W_2_bilinear = nn.Bilinear(int(self.hidden_size // 6), int(self.hidden_size // 6), 100, bias=False)
        self.W_2_linear1 = nn.Linear(int(self.hidden_size // 6), 100)
        self.W_2_linear2 = nn.Linear(int(self.hidden_size // 6), 100)

        self.bi_lism_1 = nn.LSTM(input_size=100, hidden_size=self.hidden_size//2, num_layers=1, bidirectional=True)
        self.bi_lism_2 = nn.LSTM(input_size=100, hidden_size=self.hidden_size//2, num_layers=1, bidirectional=True)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 일반화된 정보를 사용
        self.biaffine_W_bilinear = nn.Bilinear((2*(self.hidden_size//2)),(2*(self.hidden_size//2)), self.num_labels, bias=False)
        self.biaffine_W_linear1 = nn.Linear(2 * (self.hidden_size//2), self.num_labels)
        self.biaffine_W_linear2 = nn.Linear(2 * (self.hidden_size // 2), self.num_labels)
        self.reset_parameters()

    def forward(self, hidden_states, batch_size, prem_span, hypo_span):
        # hidden_states: [[batch_size, word_idxs, hidden_size], []]
        # span: [batch_size, max_sentence_length, max_sentence_length]
        # word_idxs: [batch_size, seq_length]
        # -> sequence_outputs: [batch_size, seq_length, hidden_size]

        prem_hidden_states= hidden_states[0]
        hypo_hidden_states= hidden_states[1]
        #print(prem_hidden_states.shape, hypo_hidden_states.shape, prem_span.shape, hypo_span.shape)

        # span: (batch, max_prem_len, 3) -> (batch, max_prem_len, 3*hidden_size)
        new_prem_span = torch.tensor([], dtype=torch.long).to("cuda")
        new_hypo_span = torch.tensor([], dtype=torch.long).to("cuda")

        for i, (p_span, h_span) in enumerate(zip(prem_span.tolist(), hypo_span.tolist())):
            p_span_head = torch.tensor([span[0] for span in p_span]).to("cuda") #(max_prem_len)
            p_span_tail = torch.tensor([span[1] for span in p_span]).to("cuda")
            p_span_dep = torch.tensor([span[2] for span in p_span]).to("cuda")

            p_span_head = torch.index_select(prem_hidden_states[i], 0, p_span_head) #(max_prem_len, hidden_size)
            p_span_tail = torch.index_select(prem_hidden_states[i], 0, p_span_tail)
            p_span_dep = self.depend_embedding(p_span_dep)

            n_p_span = p_span_head + p_span_tail + p_span_dep
            new_prem_span = torch.cat((new_prem_span, n_p_span.unsqueeze(0)))

            h_span_head = torch.tensor([span[0] for span in h_span]).to("cuda")  # (max_hypo_len)
            h_span_tail = torch.tensor([span[1] for span in h_span]).to("cuda")
            h_span_dep = torch.tensor([span[2] for span in h_span]).to("cuda")

            h_span_head = torch.index_select(hypo_hidden_states[i], 0, h_span_head)  # (max_hypo_len, hidden_size)
            h_span_tail = torch.index_select(hypo_hidden_states[i], 0, h_span_tail)
            h_span_dep = self.depend_embedding(h_span_dep)

            n_h_span = h_span_head + h_span_tail + h_span_dep
            new_hypo_span = torch.cat((new_hypo_span, n_h_span.unsqueeze(0)))

        prem_span = new_prem_span
        hypo_span = new_hypo_span

        del new_prem_span
        del new_hypo_span

        # biaffine attention
        # hidden_states: (batch_size, max_prem_len, hidden_size)
        # span: (batch, max_prem_len, hidden_size)
        # -> biaffine_outputs: [batch_size, max_prem_len,  100]
        prem_span = self.reduction1(prem_span)
        prem_hidden_states = self.reduction2(prem_hidden_states)
        hypo_span = self.reduction3(hypo_span)
        hypo_hidden_states = self.reduction4(hypo_hidden_states)

        prem_biaffine_outputs= self.W_1_bilinear(prem_span, prem_hidden_states) + self.W_1_linear1(prem_span) + self.W_1_linear2(prem_hidden_states)
        hypo_biaffine_outputs = self.W_2_bilinear(hypo_span, hypo_hidden_states) + self.W_2_linear1(hypo_span) + self.W_2_linear2(hypo_hidden_states)

        # bilstm
        # biaffine_outputs: [batch_size, max_sentence_length, hidden_size]
        # -> hidden_states: [batch_size, max_sentence_length]
        prem_biaffine_outputs = prem_biaffine_outputs.transpose(0,1)
        hypo_biaffine_outputs = hypo_biaffine_outputs.transpose(0,1)

        prem_bilstm_outputs, prem_states = self.bi_lism_1(prem_biaffine_outputs)
        hypo_bilstm_outputs, hypo_states = self.bi_lism_2(hypo_biaffine_outputs)

        prem_hidden_states = prem_states[0].transpose(0, 1).contiguous().view(batch_size, -1)
        hypo_hidden_states = hypo_states[0].transpose(0, 1).contiguous().view(batch_size, -1)

        # biaffine attention
        # prem_hidden_states: (batch_size, max_prem_len)
        # hypo_hidden_states: (batch_size, max_hypo_len)
        # -> outputs: (batch_size, num_labels)
        outputs = self.biaffine_W_bilinear(prem_hidden_states, hypo_hidden_states) + self.biaffine_W_linear1(prem_hidden_states) +self.biaffine_W_linear2(hypo_hidden_states)

        return outputs

    def reset_parameters(self):
        self.W_1_bilinear.reset_parameters()
        self.W_1_linear1.reset_parameters()
        self.W_1_linear2.reset_parameters()
        self.W_2_bilinear.reset_parameters()
        self.W_2_linear1.reset_parameters()
        self.W_2_linear2.reset_parameters()

        self.biaffine_W_bilinear.reset_parameters()
        self.biaffine_W_linear1.reset_parameters()
        self.biaffine_W_linear2.reset_parameters()
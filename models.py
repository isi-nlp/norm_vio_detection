import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BERTRNN(nn.Module):
    def __init__(self, model_name, n_rnn_layers, dropout=0.5, n_finetune_layers=0):
        super(BERTRNN, self).__init__()

        if model_name in ['bert-base-uncased', 'gpt2']:
            from transformers import AutoModel
            self.bert = AutoModel.from_pretrained(model_name)
        else:
            from transformers import T5EncoderModel
            self.bert = T5EncoderModel.from_pretrained(model_name)
        nhid = self.bert.config.hidden_size

        # only finetune the top several layers of BERT

        n_layers = 12
        if n_finetune_layers > 0:
            for param in self.bert.parameters():
                param.requires_grad = False
            if model_name == 'bert-base-uncased':
                for param in self.bert.pooler.parameters():
                    param.requires_grad = True
                for i in range(n_layers - 1, n_layers - 1 - n_finetune_layers, -1):
                    for param in self.bert.encoder.layer[i].parameters():
                        param.requires_grad = True
            elif model_name == 'gpt2':
                for param in self.bert.ln_f.parameters():
                    param.requires_grad = True
                for i in range(n_layers - 1, n_layers - 1 - n_finetune_layers, -1):
                    for param in self.bert.h[i].parameters():
                        param.requires_grad = True
            elif model_name == 't5-base':
                for param in self.bert.encoder.final_layer_norm.parameters():
                    param.requires_grad = True
                for i in range(n_layers - 1, n_layers - 1 - n_finetune_layers, -1):
                    for param in self.bert.encoder.block[i].parameters():
                        param.requires_grad = True

        self.rnn = nn.GRU(nhid, nhid // 2, num_layers=n_rnn_layers, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(nhid, 2)
        self.dropout = nn.Dropout(p=dropout)
        self.model_name = model_name

    def forward(self, input_ids, attention_mask, conv_lens):
        conv_lens = conv_lens.to('cpu')
        batch_size, max_conv_len, seq_len = input_ids.shape
        # import ipdb; ipdb.set_trace()

        input_ids = input_ids.reshape(-1, seq_len)  # (batch_size*max_conv_len, seq_len)
        attention_mask = attention_mask.reshape(-1, seq_len)    # (batch_size*max_conv_len, seq_len)

        outputs = self.bert(input_ids, attention_mask=attention_mask)
        if self.model_name == 'bert-base-uncased':
            last_hidden_states = outputs.pooler_output   # (batch_size*max_conv_len, hidden_size)
        else:
            last_hidden_states = outputs.last_hidden_state  # (batch_size*max_conv_len, seq_len, hidden_size)
            attention_mask = attention_mask.to(torch.float)
            attention_mask = attention_mask / attention_mask.sum(dim=1, keepdims=True)
            last_hidden_states = torch.einsum('abc,ab->ac', last_hidden_states, attention_mask)  # (batch_size*max_conv_len, hidden_size)
        last_hidden_states = last_hidden_states.reshape(batch_size, max_conv_len, -1)   # (batch_size, max_conv_len, hidden_size)
        hidden_size = last_hidden_states.shape[-1]

        # reshape BERT embeddings to fit into RNN
        last_hidden_states = last_hidden_states.permute(1, 0, 2)  # (max_conv_len, batch_size, hidden_size)

        # sequence modeling of act tags using RNN
        last_hidden_states = pack_padded_sequence(last_hidden_states, conv_lens, enforce_sorted=False)
        self.rnn.flatten_parameters()
        outputs, _ = self.rnn(last_hidden_states)
        outputs, _ = pad_packed_sequence(outputs)  # (batch_max_conv_len, batch_size, hidden_size)
        if outputs.shape[0] < max_conv_len:
            outputs_padding = torch.zeros(max_conv_len - outputs.shape[0], batch_size, hidden_size, device=outputs.device)
            outputs = torch.cat([outputs, outputs_padding], dim=0)  # (max_conv_len, batch_size, hidden_size)
        outputs = self.dropout(outputs)
        outputs = self.fc(outputs)  # (max_conv_len, batch_size, 2)
        outputs = outputs.permute(1, 0, 2)  # (batch_size, max_conv_len, 2)
        print('**********', conv_lens, '**********')
        last_cell_indices = [list(range(batch_size)), (conv_lens-1).numpy().tolist()]
        outputs = outputs[last_cell_indices]  # (batch_size, 2)

        return outputs


class BERT(nn.Module):
    def __init__(self, model_name, dropout=0.5, n_finetune_layers=0):
        super(BERT, self).__init__()

        if model_name in ['bert-base-uncased', 'gpt2']:
            from transformers import AutoModel
            self.bert = AutoModel.from_pretrained(model_name)
        else:
            from transformers import T5EncoderModel
            self.bert = T5EncoderModel.from_pretrained(model_name)
        nhid = self.bert.config.hidden_size

        # only finetune the top several layers of BERT
        n_layers = 12
        if n_finetune_layers > 0:
            print('Freezing some top layers....')
            for param in self.bert.parameters():
                param.requires_grad = False
            if model_name == 'bert-base-uncased':
                for param in self.bert.pooler.parameters():
                    param.requires_grad = True
                for i in range(n_layers - 1, n_layers - 1 - n_finetune_layers, -1):
                    for param in self.bert.encoder.layer[i].parameters():
                        param.requires_grad = True
            elif model_name == 'gpt2':
                for param in self.bert.ln_f.parameters():
                    param.requires_grad = True
                for i in range(n_layers - 1, n_layers - 1 - n_finetune_layers, -1):
                    for param in self.bert.h[i].parameters():
                        param.requires_grad = True
            elif model_name == 't5-base':
                for param in self.bert.encoder.final_layer_norm.parameters():
                    param.requires_grad = True
                for i in range(n_layers - 1, n_layers - 1 - n_finetune_layers, -1):
                    for param in self.bert.encoder.block[i].parameters():
                        param.requires_grad = True

        self.fc = nn.Linear(nhid, 2)
        self.dropout = nn.Dropout(p=dropout)
        self.model_name = model_name

    def forward(self, input_ids, attention_mask):
        batch_size, max_conv_len, seq_len = input_ids.shape

        input_ids = input_ids.reshape(-1, seq_len)  # (batch_size, seq_len)
        attention_mask = attention_mask.reshape(-1, seq_len)    # (batch_size, seq_len)

        outputs = self.bert(input_ids, attention_mask=attention_mask)
        if self.model_name == 'bert-base-uncased':
            last_hidden_states = outputs.pooler_output   # (batch_size, hidden_size)
        else:
            last_hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
            attention_mask = attention_mask.to(torch.float)
            attention_mask = attention_mask / attention_mask.sum(dim=1, keepdims=True)
            last_hidden_states = torch.einsum('abc,ab->ac', last_hidden_states, attention_mask)  # (batch_size, hidden_size)

        outputs = self.dropout(last_hidden_states)
        outputs = self.fc(outputs)  # (batch_size, 2)
        return outputs

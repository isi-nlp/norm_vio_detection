import torch
import torch.nn as nn
import os
import numpy as np
from dataset import data_loader, all_cats, data_loader_inference
from models import BERTRNN
from sklearn.metrics import f1_score
from collections import OrderedDict


class Evaluator:
    def __init__(self, args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.device_count() > 0:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

        print('Done\n')

        os.makedirs('ckps', exist_ok=True)

        print('Initializing model....')
        model = BERTRNN(
            n_rnn_layers=args.n_rnn_layers,
            dropout=args.dropout,
            n_finetune_layers=0
        )

        print('Resuming from the saved checkpoint....')
        prefix = f'ckps/{args.model_name}'
        state_dict = torch.load(f'{prefix}/model_{args.idx}.pt', map_location=device)
        for each in state_dict:
            state_dict[each] = state_dict[each].to(device)

        model = nn.DataParallel(model)
        model.load_state_dict(state_dict)
        model.to(device)
        print('Done\n')

        self.device = device
        self.model = model
        self.args = args

    def eval(self, val=True):
        loader = data_loader(model_name=self.args.model_name,
                               phase='dev' if val else 'test',
                               batch_size=self.args.batch_size,
                               max_context_size=self.args.max_context_size,
                               max_n_tokens=self.args.max_n_tokens,
                               n_workers=self.args.n_workers)

        print('Inferencing....')
        y_pred = []
        y_true = []
        cats = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                batch_input_ids = batch['input_ids'].to(self.device)
                batch_attention_mask = batch['attention_mask'].to(self.device)
                batch_conv_lens = batch['conv_len'].to(self.device)
                batch_labels = batch['label']
                batch_cats = batch['cat']
                logits = self.model(batch_input_ids, batch_attention_mask, batch_conv_lens)
                preds = logits.detach().to('cpu').argmax(dim=1).numpy()
                y_pred.append(preds)
                y_true.append(batch_labels.numpy())
                cats.extend(batch_cats)
        print('Done\n')

        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        cats = np.array(cats)

        cat2f1 = OrderedDict({'average': 0})
        f1s = []
        for cat in all_cats:
            mask_cat = cats == cat
            f1_cat = f1_score(y_true[mask_cat], y_pred[mask_cat])
            cat2f1[cat] = f1_cat
            f1s.append(f1_cat)
        average_f1 = sum(f1s) / len(f1s)
        cat2f1['average'] = average_f1

        return cat2f1

    def inference(self, conversations, subreddits, rules):
        loader = data_loader_inference(self.args.model_name,
                                       conversations=conversations,
                                       subreddits=subreddits,
                                       rules=rules,
                                       batch_size=self.args.batch_size,
                                       max_context_size=self.args.max_context_size,
                                       max_n_tokens=self.args.max_n_tokens,
                                       n_workers=self.args.n_workers)

        # make sure all conversations are at least of length 2 (one context comment and one target comment)
        for conv in conversations:
            if len(conv) == 1:
                conv.insert(0, 'none')

        print('Inferencing....')
        probs_ = []
        softmax = nn.Softmax(dim=-1)
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                batch_input_ids = batch['input_ids'].to(self.device)
                batch_attention_mask = batch['attention_mask'].to(self.device)
                batch_conv_lens = batch['conv_len'].to(self.device)
                logits = self.model(batch_input_ids, batch_attention_mask, batch_conv_lens)
                probs = softmax(logits.detach()).to('cpu').numpy()[:, 1]
                probs_.append(probs)
        print('Done\n')

        probs_ = np.concatenate(probs_, axis=0).tolist()
        return probs_

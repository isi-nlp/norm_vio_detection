import pandas as pd
import torch
import torch.nn as nn
import os
import numpy as np
from dataset import data_loader, all_cats
from models import BERTRNN, BERT
import copy
import time
from sklearn.metrics import f1_score
from collections import OrderedDict


class Trainer:
    def __init__(self, args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.device_count() > 0:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            print(torch.cuda.get_device_properties(0))

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

        print('Preparing datasets....')
        train_loader = data_loader(phase='train',
                                   model_name=args.model_name,
                                   batch_size=args.batch_size,
                                   sample_rate=args.sample_rate_train,
                                   n_few_shot=args.n_few_shot,
                                   cat=args.src_cat,
                                   comm=args.src_comm,
                                   max_context_size=args.max_context_size,
                                   max_n_tokens=args.max_n_tokens,
                                   n_workers=args.n_workers,
                                   seed=args.seed)

        val_loader = data_loader(phase='dev',
                                 model_name=args.model_name,
                                 batch_size=args.batch_size,
                                 sample_rate=args.sample_rate_train,
                                 n_few_shot=args.n_few_shot,
                                 cat=args.src_cat,
                                 comm=args.src_comm,
                                 max_context_size=args.max_context_size,
                                 max_n_tokens=args.max_n_tokens,
                                 n_workers=args.n_workers,
                                 seed=args.seed)

        test_loader = data_loader(phase='test',
                                  model_name=args.model_name,
                                  batch_size=args.batch_size,
                                  sample_rate=args.sample_rate_train,
                                  n_few_shot=args.n_few_shot,
                                  cat=args.tgt_cat,
                                  comm=args.tgt_comm,
                                  max_context_size=args.max_context_size,
                                  max_n_tokens=args.max_n_tokens,
                                  n_workers=args.n_workers,
                                  seed=args.seed)

        print('Done\n')

        print('Initializing model....')
        if args.max_context_size > 0:
            model = BERTRNN(
                model_name=args.model_name,
                n_rnn_layers=args.n_rnn_layers,
                dropout=args.dropout,
                n_finetune_layers=args.n_finetune_layers
            )
        else:
            model = BERT(
                model_name=args.model_name,
                dropout=args.dropout
            )

        model.to(device)

        from transformers import AdamW
        params = model.parameters()
        optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)

        prefix = f'ckps/{args.task}/{args.model_name}/{args.idx}/seed={args.seed}'
        os.makedirs(prefix, exist_ok=True)

        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.prefix = prefix

    def train(self):
        best_epoch = 0
        best_epoch_f1 = 0
        best_state_dict = copy.deepcopy(self.model.state_dict())

        t = time.time()
        for epoch in range(self.args.epochs):
            print(f"{'*' * 20}Epoch: {epoch + 1}{'*' * 20}")
            loss = self.train_epoch()
            cat2f1 = self.eval()
            f1 = cat2f1['macro']
            if f1 > best_epoch_f1:
                best_epoch = epoch
                best_epoch_f1 = f1
                best_state_dict = copy.deepcopy(self.model.state_dict())
            print(f'Epoch {epoch + 1}\tTrain Loss: {loss:.3f}\tVal F1: {f1:.3f}\n'
                  f'Best Epoch: {best_epoch + 1}\tBest Epoch Val F1: {best_epoch_f1:.3f}\n\n'
                  )
            for cat in cat2f1:
                f1 = cat2f1[cat]
                print(f'Val F1_{cat}: {f1:.3f}')
            print()
            if epoch - best_epoch >= 10:
                break

        elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - t))
        print(f'Elapsed Time: {elapsed_time}')
        print('Saving the best checkpoint....')
        torch.save(best_state_dict, f"{self.prefix}/model.pt")
        self.model.load_state_dict(best_state_dict)
        cat2f1 = self.eval(False)

        for cat in cat2f1:
            f1 = cat2f1[cat]
            print(f'Test F1_{cat}: {f1:.3f}')

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        t_epoch = time.time()
        t = time.time()
        for i, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            batch_input_ids = batch['input_ids'].to(self.device)
            batch_attention_mask = batch['attention_mask'].to(self.device)
            batch_conv_lens = batch['conv_len'].to(self.device)
            batch_labels = batch['label'].to(self.device)
            if self.args.max_context_size > 0:
                logits = self.model(batch_input_ids, batch_attention_mask, batch_conv_lens)
            else:
                logits = self.model(batch_input_ids, batch_attention_mask)
            loss = self.criterion(logits, batch_labels)
            loss.backward()
            self.optimizer.step()
            interval = max(len(self.train_loader) // 20, 1)
            batch_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - t))
            if i % interval == 0 or i == len(self.train_loader) - 1:
                print(f'Batch: {i + 1}/{len(self.train_loader)}\tloss: {loss.item():.3f}\tbatch_time: {batch_time}')
                t = time.time()
            epoch_loss += loss.item()
        epoch_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - t_epoch))
        print(f"Epoch training time: {epoch_time}\n")
        return epoch_loss / len(self.train_loader)

    def eval(self, val=True):
        y_pred = []
        y_true = []
        cats = []
        loader = self.val_loader if val else self.test_loader
        print('Inferencing....')
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                batch_input_ids = batch['input_ids'].to(self.device)
                batch_attention_mask = batch['attention_mask'].to(self.device)
                batch_conv_lens = batch['conv_len'].to(self.device)
                batch_labels = batch['label']
                batch_cats = batch['cat']
                if self.args.max_context_size > 0:
                    logits = self.model(batch_input_ids, batch_attention_mask, batch_conv_lens)
                else:
                    logits = self.model(batch_input_ids, batch_attention_mask)
                preds = logits.detach().to('cpu').argmax(dim=1).numpy()
                y_pred.append(preds)
                y_true.append(batch_labels.numpy())
                cats.extend(batch_cats)
        print('Done\n')

        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        cats = np.array(cats)
        cats = pd.Series(cats)

        f1_micro = f1_score(y_true, y_pred)
        cat2f1 = OrderedDict({'macro': 0, 'micro': f1_micro})
        f1s = []
        for cat in all_cats:
            mask_cat = cats.apply(lambda x: cat in x).values
            if mask_cat.sum() != 0:
                f1_cat = f1_score(y_true[mask_cat], y_pred[mask_cat])
                cat2f1[cat] = f1_cat
                f1s.append(f1_cat)

        cat = self.args.src_cat if val else self.args.tgt_cat
        if cat == 'all':
            average_f1 = sum(f1s) / len(f1s)
        elif cat[0] != '~':
            average_f1 = cat2f1[cat]
        else:
            average_f1 = sum(f1s) / len(f1s)

        cat2f1['macro'] = average_f1
        return cat2f1

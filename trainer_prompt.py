import torch
import torch.nn as nn
import os
import numpy as np
from dataset import all_cats, create_normvio_prompt_dataset
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
        from openprompt.plms import load_plm
        plm, tokenizer, model_config, WrapperClass = load_plm(args.model_name.split('-')[0], args.model_name)

        from openprompt.prompts import ManualTemplate
        template = "In the {'meta':'subreddit'} subreddit, there is a rule: {'meta':'rule'}. "
        template += "A conversation took place: "
        for i in range(args.max_context_size):
            template += f"Comment {i + 1}: {{'meta': 'comment{i}', 'shortenable': True}}\n"
        if args.max_context_size > 0:
            template += f"Comment {args.max_context_size + 1}: {{'meta': 'comment{args.max_context_size}', 'shortenable': True}}\n"
            template += "Does the last comment violate the subreddit rule? (yes/no) {'mask'}"
        else:
            template += f"Comment: {{'meta': 'comment{args.max_context_size}', 'shortenable': True}}\n"
            template += "Does the comment violate the subreddit rule? (yes/no) {'mask'}"

        mytemplate = ManualTemplate(tokenizer=tokenizer, text=template)
        data_train, features_train = create_normvio_prompt_dataset(phase='train',
                                                                   sample_rate=args.sample_rate_train,
                                                                   n_few_shot=args.n_few_shot,
                                                                   cat=args.src_cat,
                                                                   comm=args.src_comm,
                                                                   max_context_size=args.max_context_size,
                                                                   seed=args.seed)
        data_val, features_val = create_normvio_prompt_dataset(phase='dev',
                                                               sample_rate=args.sample_rate_train,
                                                               n_few_shot=args.n_few_shot,
                                                               cat=args.src_cat,
                                                               comm=args.src_comm,
                                                               max_context_size=args.max_context_size,
                                                               seed=args.seed)
        data_test, features_test = create_normvio_prompt_dataset(phase='test',
                                                                 sample_rate=args.sample_rate_train,
                                                                 n_few_shot=args.n_few_shot,
                                                                 cat=args.tgt_cat,
                                                                 comm=args.tgt_comm,
                                                                 max_context_size=args.max_context_size,
                                                                 seed=args.seed)

        from openprompt import PromptDataLoader
        train_loader = PromptDataLoader(dataset=data_train, template=mytemplate, tokenizer=tokenizer,
                                        tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_n_tokens,
                                        decoder_max_length=3, batch_size=args.batch_size, shuffle=True,
                                        teacher_forcing=False,
                                        predict_eos_token=False, truncate_method="tail")
        val_loader = PromptDataLoader(dataset=data_val, template=mytemplate, tokenizer=tokenizer,
                                      tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_n_tokens,
                                      decoder_max_length=3, batch_size=args.batch_size, shuffle=False,
                                      teacher_forcing=False,
                                      predict_eos_token=False, truncate_method="tail")
        test_loader = PromptDataLoader(dataset=data_test, template=mytemplate, tokenizer=tokenizer,
                                       tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_n_tokens,
                                       decoder_max_length=3, batch_size=args.batch_size, shuffle=False,
                                       teacher_forcing=False,
                                       predict_eos_token=False, truncate_method="tail")

        print('Done\n')

        print('Initializing model....')
        from openprompt.prompts import ManualVerbalizer
        myverbalizer = ManualVerbalizer(tokenizer, num_classes=2, label_words=["no", "yes"])
        from openprompt import PromptForClassification
        model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'***********{n_params} trainable parameters!***********')

        n_layers = 12
        if args.n_finetune_layers > 0:
            print('Freezing some layers....\n')
            for param in plm.parameters():
                param.requires_grad = False
            for param in model.verbalizer.parameters():
                param.requires_grad = True
            for param in model.prompt_model.template.parameters():
                param.requires_grad = True

            if args.model_name == 'bert-base-uncased':
                for param in plm.cls.parameters():
                    param.requires_grad = True
                for i in range(n_layers - 1, n_layers - 1 - args.n_finetune_layers, -1):
                    for param in plm.bert.encoder.layer[i].parameters():
                        param.requires_grad = True
            elif args.model_name == 'gpt2':
                for param in plm.lm_head.parameters():
                    param.requires_grad = True
                for param in plm.transformer.ln_f.parameters():
                    param.requires_grad = True
                for i in range(n_layers - 1, n_layers - 1 - args.n_finetune_layers, -1):
                    for param in plm.transformer.h[i].parameters():
                        param.requires_grad = True
            elif args.model_name == 't5-base':
                for param in plm.lm_head.parameters():
                    param.requires_grad = True
                for param in plm.decoder.final_layer_norm.parameters():
                    param.requires_grad = True
                for i in range(n_layers - 1, n_layers - 1 - args.n_finetune_layers, -1):
                    for param in plm.decoder.block[i].parameters():
                        param.requires_grad = True

        model.to(device)
        print('Done\n')

        from transformers import AdamW
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
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
        self.features_train = features_train
        self.features_val = features_val
        self.features_test = features_test
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
            batch = batch.to(self.device)
            batch_labels = batch['label']
            logits = self.model(batch)
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
        loader = self.val_loader if val else self.test_loader
        features = self.features_val if val else self.features_test
        print('Inferencing....')
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                batch = batch.to(self.device)
                batch_labels = batch['label']
                logits = self.model(batch)
                preds = logits.detach().to('cpu').argmax(dim=1).numpy()
                y_pred.append(preds)
                y_true.append(batch_labels.to('cpu').numpy())
        print('Done\n')

        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        cats = features

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

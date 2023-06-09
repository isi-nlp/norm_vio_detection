import torch
import torch.nn as nn
import os
import numpy as np
from dataset_inference import data_loader
from models import BERTRNN


class Evaluator:
    def __init__(self, args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.device_count() > 0:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            print(torch.cuda.get_device_properties(0))

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

        os.makedirs('ckps', exist_ok=True)

        print('Initializing model....')
        model = BERTRNN(
            model_name=args.model_name,
            n_rnn_layers=args.n_rnn_layers,
            dropout=args.dropout,
        )

        print('Resuming from the saved checkpoint....')
        prefix = f'ckps/{args.task}/{args.model_name}/{args.idx}/seed=2022'
        state_dict = torch.load(f'{prefix}/model.pt', map_location=device)
        for each in state_dict:
            state_dict[each] = state_dict[each].to(device)
        model.load_state_dict(state_dict)
        model.to(device)
        print('Done\n')

        self.device = device
        self.model = model
        self.args = args

    def inference(self, conversations, subreddits, rules):
        loader = data_loader(conversations=conversations,
                             subreddits=subreddits,
                             rules=rules,
                             batch_size=self.args.batch_size,
                             model_name=self.args.model_name,
                             max_context_size=self.args.max_context_size,
                             max_n_tokens=self.args.max_n_tokens,
                             n_workers=self.args.n_workers
                             )

        # make sure all conversations are at least of length 2 (one context comment and one target comment)
        for conv in conversations:
            if len(conv) == 1:
                conv.insert(0, 'None.')

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

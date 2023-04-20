import os
from trainer import Trainer
import argparse
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, default=1)
    parser.add_argument('--model_name', type=str, choices=('BERTRNN', 'GPT2'), default='BERTRNN')

    # data config
    parser.add_argument('--max_context_size', type=int, default=5)
    parser.add_argument('--max_n_tokens', type=int, default=128)

    # model config
    parser.add_argument('--n_rnn_layers', type=int, default=2)
    parser.add_argument('--dropout', type=int, default=0.5)
    parser.add_argument('--n_finetune_layers', type=int, default=0)

    # training config
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--n_workers', type=int, default=2)
    parser.add_argument('--gpu', type=str, default='')
    parser.add_argument('--seed', type=int, default=2022)

    args = parser.parse_args()
    print(args)

    prefix = f'results/{args.model_name}/{args.idx}'

    args_dict = args.__dict__
    with open(f'{prefix}/config.json', 'w') as f:
        json.dump(args_dict, f, indent=2)
    print(json.dumps(args_dict, indent=2))

    trainer = Trainer(args)
    trainer.train()
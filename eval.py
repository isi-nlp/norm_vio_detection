import argparse
import json
from evaluator import Evaluator


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, choices=('BERTRNN', 'GPT2'), default='BERTRNN')
    parser.add_argument('--idx', type=int, default=1)

    # data config
    parser.add_argument('--max_context_size', type=int, default=5)
    parser.add_argument('--max_n_tokens', type=int, default=128)

    # model config
    parser.add_argument('--n_rnn_layers', type=int, default=2)

    # training config
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--gpu', type=str, default='')
    parser.add_argument('--seed', type=int, default=2022)

    args = parser.parse_args()

    prefix = f'results/{args.idx}'
    with open(f'{prefix}/config.json', 'w') as f:
        config = json.load(f)
    for name in ['max_context_size', 'max_n_tokens', 'n_rnn_layers']:
        args.__setattr__(name, config[name])

    print(json.dumps(args.__dict__, indent=2))

    engine = Evaluator(args)
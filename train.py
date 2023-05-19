import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, default=1)
    parser.add_argument('--task', type=str, choices=('clf', 'prompt'), default='clf')
    parser.add_argument('--model_name', type=str, choices=('bert-base-uncased', 'gpt2', 't5-base'), default='bert-base')

    # data config
    parser.add_argument('--max_context_size', type=int, default=5)
    parser.add_argument('--max_n_tokens', type=int, default=128)
    parser.add_argument('--sample_rate_train', type=float, default=1.)
    parser.add_argument('--src_cat', type=str, default='all',
                        choices=('all', 'incivility', 'harassment', 'spam', 'format', 'content',
                                 'off-topic', 'hatespeech', 'trolling', 'meta-rules',
                                 '~incivility', '~harassment', '~spam', '~format', '~content',
                                 '~off-topic', '~hatespeech', '~trolling', '~meta-rules',
                                 ))
    parser.add_argument('--tgt_cat', type=str, default='all',
                        choices=('all', 'incivility', 'harassment', 'spam', 'format', 'content',
                                 'off-topic', 'hatespeech', 'trolling', 'meta-rules'))
    parser.add_argument('--src_comm', type=str, default='all',
                        choices=('all', 'Coronavirus', 'CanadaPolitics', 'LabourUK', 'TexasPolitics',
                                 'classicwow', 'Games', 'RPClipsGTA', 'heroesofthestorm',
                                 '~Coronavirus', '~CanadaPolitics', '~LabourUK', '~TexasPolitics',
                                 '~classicwow', '~Games', '~RPClipsGTA', '~heroesofthestorm'
                                 )
                        )
    parser.add_argument('--tgt_comm', type=str, default='all',
                        choices=('all', 'Coronavirus', 'CanadaPolitics', 'LabourUK', 'TexasPolitics',
                                 'classicwow', 'Games', 'RPClipsGTA', 'heroesofthestorm')
                        )
    parser.add_argument('--n_few_shot', type=int, default=0)

    # clf model config
    parser.add_argument('--n_rnn_layers', type=int, default=2)
    parser.add_argument('--dropout', type=int, default=0.5)

    # training config
    parser.add_argument('--n_finetune_layers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--n_workers', type=int, default=2)
    parser.add_argument('--gpu', type=str, default='')
    parser.add_argument('--seed', type=int, default=2022)

    args = parser.parse_args()
    print(args)

    prefix = f'results/{args.task}/{args.model_name}/{args.idx}/seed={args.seed}'

    args_dict = args.__dict__
    with open(f'{prefix}/config.json', 'w') as f:
        json.dump(args_dict, f, indent=2)
    print(json.dumps(args_dict, indent=2))

    if args.task == 'clf':
        from trainer import Trainer
    else:
        from trainer_prompt import Trainer
    trainer = Trainer(args)
    trainer.train()

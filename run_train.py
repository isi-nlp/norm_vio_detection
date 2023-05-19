import os
import glob
import re

if __name__ == '__main__':
    for seed in [
        # 2022,
        # 2021,
        2020,
    ]:
        task = ['clf', 'prompt'][1]
        model_name = ['bert-base-uncased', 'gpt2', 't5-base'][2]

        src_cat = ['all', 'incivility', 'harassment', 'spam', 'format', 'content',
                   'off-topic', 'hatespeech', 'trolling', 'meta-rules',
                   '~incivility', '~harassment', '~spam', '~format', '~content',
                   '~off-topic', '~hatespeech', '~trolling', '~meta-rules',
                   ][-1]
        tgt_cat = ['all', 'incivility', 'harassment', 'spam', 'format', 'content',
                   'off-topic', 'hatespeech', 'trolling', 'meta-rules'][9]
        src_comm = ['all', 'Coronavirus', 'CanadaPolitics', 'LabourUK', 'TexasPolitics',
                    'classicwow', 'Games', 'RPClipsGTA', 'heroesofthestorm',
                    '~Coronavirus', '~CanadaPolitics', '~LabourUK', '~TexasPolitics',
                    '~classicwow', '~Games', '~RPClipsGTA', '~heroesofthestorm'
                    ][0]
        tgt_comm = ['all', 'Coronavirus', 'CanadaPolitics', 'LabourUK', 'TexasPolitics',
                    'classicwow', 'Games', 'RPClipsGTA', 'heroesofthestorm'][0]
        n_few_shot = [0, 10, 50, 100][0]

        sample_rate_train = 1.

        max_context_size = 5
        max_n_tokens = {'clf': 128, 'prompt': 384}[task]

        batch_size = {
            'clf': {'bert-base-uncased': 224, 'gpt2': 96, 't5-base': 200},
            'prompt': {'bert-base-uncased': 54, 'gpt2': 36, 't5-base': 44}
        }[task][model_name]

        epochs = 50
        lr = 1e-4
        weight_decay = 5e-4
        n_rnn_layers = 2
        n_finetune_layers = {'clf': 2, 'prompt': 0}[task]
        n_workers = 4

        idx = 20
        # indices = []
        # folders = glob.glob(f'results/{task}/{model_name}/*')
        # for folder in folders:
        #     idx = int(re.findall(r'\d+', folder)[-1])
        #     indices.append(idx)
        # if seed == 2022:
        #     idx = max(indices) + 1 if indices else 1
        # else:
        #     idx = max(indices) if indices else 1

        prefix = f'results/{task}/{model_name}/{idx}/seed={seed}'
        os.makedirs(prefix, exist_ok=True)

        gpu = '0'
        gpu_type = ['p100', 'v100', 'a100', 'a40'][1]
        time = {'bert-base-uncased': '20:00:00',
                'gpt2': '20:00:00',
                't5-base': '20:00:00'
                }[model_name]

        print(f'**********Requested time: {time}*************')

        append = '>'

        command = f'python -u train.py ' \
                  f'--task={task} ' \
                  f'--model_name={model_name} ' \
                  f'--sample_rate_train={sample_rate_train} ' \
                  f'--n_few_shot={n_few_shot} ' \
                  f'--src_cat={src_cat} ' \
                  f'--tgt_cat={tgt_cat} ' \
                  f'--src_comm={src_comm} ' \
                  f'--tgt_comm={tgt_comm} ' \
                  f'--max_context_size={max_context_size} ' \
                  f'--max_n_tokens={max_n_tokens} ' \
                  f'--n_rnn_layers={n_rnn_layers} ' \
                  f'--n_finetune_layers={n_finetune_layers} ' \
                  f'--idx={idx} ' \
                  f'--seed={seed} ' \
                  f'--batch_size={batch_size} ' \
                  f'--epochs={epochs} ' \
                  f'--lr={lr} ' \
                  f'--weight_decay={weight_decay} ' \
                  f'--gpu={gpu} ' \
                  f'--n_workers={n_workers} ' \
                  f'{append} {prefix}/output.txt'

        print(command)
        # os.system(command)

        n_gpus = len(gpu.split(','))
        script = f"""#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --cpus-per-task={n_workers}
#SBATCH --gres=gpu:{gpu_type}:{n_gpus}
#SBATCH --mem=16GB
#SBATCH --time={time}

{command}
"""
        with open(f'run_slurm.sh', 'w') as f:
            f.write(script)
        print()
        print('This is the script.')
        print(script)
        os.system(f'sbatch run_slurm.sh')

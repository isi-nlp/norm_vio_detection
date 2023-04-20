import os

if __name__ == '__main__':
    model_name = ['BERTRNN', 'GPT'][0]
    idx = 1

    max_context_size = 5
    max_n_tokens = 128

    batch_size = 64
    epochs = 50
    lr = 1e-4
    weight_decay = 5e-4
    n_rnn_layers = 2
    n_finetune_layers = 2
    n_workers = 4

    prefix = f'results/{model_name}/{idx}'
    os.makedirs(prefix, exist_ok=True)

    gpu = '1'
    gpu_type = ['p100', 'v100', 'a100', 'a40'][2]
    time = {'BERTRNN': '28:00:00',
            'GPT': '28:00:00'
            }[model_name]

    print(f'**********Requested time: {time}*************')

    append = '>'

    command = f'python -u train.py ' \
              f'--model_name={model_name} ' \
              f'--max_context_size={max_context_size} ' \
              f'--max_n_tokens={max_n_tokens} ' \
              f'--n_rnn_layers={n_rnn_layers} ' \
              f'--n_finetune_layers={n_finetune_layers} ' \
              f'--idx={idx} ' \
              f'--batch_size={batch_size} ' \
              f'--epochs={epochs} ' \
              f'--lr={lr} ' \
              f'--weight_decay={weight_decay} ' \
              f'--gpu={gpu} ' \
              f'--n_workers={n_workers} ' \
              # f'{append} {prefix}/output.txt'

    print(command)
    os.system(command)

#     n_gpus = len(gpu.split(','))
#     script = f"""#!/bin/bash
# #SBATCH --partition=gpu
# #SBATCH --nodes=1
# #SBATCH --ntasks=1
# #SBATCH --cpus-per-task={n_workers}
# #SBATCH --gres=gpu:{gpu_type}:{n_gpus}
# #SBATCH --mem=16GB
# #SBATCH --time={time}
#
#     {command}
#                 """
#     with open(f'run_slurm.sh', 'w') as f:
#         f.write(script)
#     print()
#     print('This is the script.')
#     print(script)
#     os.system(f'sbatch run_slurm.sh')
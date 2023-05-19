import torch
import torch.nn as nn
import os
import numpy as np
from dataset_inference import data_loader, create_normvio_prompt_dataset
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

        print('Initializing model....')
        from openprompt.prompts import ManualVerbalizer
        myverbalizer = ManualVerbalizer(tokenizer, num_classes=2, label_words=["no", "yes"])
        from openprompt import PromptForClassification
        model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'***********{n_params} trainable parameters!***********')

        print('Resuming from the saved checkpoint....')
        prefix = f'ckps/{args.task}/{args.model_name}/{args.idx}/seed=2022'
        print(os.path.exists(f'{prefix}/model.pt'))
        state_dict = torch.load(f'{prefix}/model.pt', map_location=device)
        for each in state_dict:
            state_dict[each] = state_dict[each].to(device)
        model.load_state_dict(state_dict)
        model.to(device)
        print('Done\n')

        self.device = device
        self.model = model
        self.template = mytemplate
        self.tokenizer = tokenizer
        self.wrapper_class = WrapperClass
        self.args = args

    def inference(self, conversations, subreddits, rules):
        data = create_normvio_prompt_dataset(
            conversations=conversations,
            subreddits=subreddits,
            rules=rules,
            max_context_size=self.args.max_context_size
        )

        from openprompt import PromptDataLoader
        loader = PromptDataLoader(dataset=data, template=self.template, tokenizer=self.tokenizer,
                                  tokenizer_wrapper_class=self.wrapper_class,
                                  max_seq_length=self.args.max_n_tokens, decoder_max_length=3,
                                  batch_size=self.args.batch_size, shuffle=False, teacher_forcing=False,
                                  predict_eos_token=False, truncate_method="tail")

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
                batch = batch.to(self.device)
                logits = self.model(batch)
                probs = softmax(logits.detach()).to('cpu').numpy()[:, 1]
                probs_.append(probs)
        print('Done\n')

        probs_ = np.concatenate(probs_, axis=0).tolist()
        return probs_

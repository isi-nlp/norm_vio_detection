import pandas as pd
import torch
import itertools
from torch.utils.data import Dataset, DataLoader
import redditcleaner
import preprocessor as p

p.set_options(p.OPT.URL, p.OPT.EMOJI)

all_cats = ['incivility', 'harassment', 'spam', 'format', 'content',
            'off-topic', 'hatespeech', 'trolling', 'meta-rules']


class NormVioSeqInference(Dataset):
    def __init__(self, conversations, subreddits, rules, model_name='bert-base-uncased',
                 max_context_size=5, max_n_tokens=128, n_workers=4):
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=n_workers, progress_bar=False)

        last_comments = [conv[-1] for conv in conversations]
        contexts = [conv[:-1] for conv in conversations]
        df = pd.DataFrame({
            'final_comment': last_comments,
            'context': contexts,
            'subreddit': subreddits,
            'rule_texts': rules
        })
        n = df.shape[0]

        def truncate_context(x):
            # only keep a few predecessors
            x = x[-max_context_size:]
            return x
        df['context'] = df['context'].parallel_apply(truncate_context)

        def reddit_clean(x):
            return p.tokenize(redditcleaner.clean(x))

        def reddit_batch_clean(x):
            return [p.tokenize(redditcleaner.clean(e)) for e in x]

        def augment_comment(row):
            comment = row['final_comment']
            subrredit = row['subreddit']
            rule_text = row['rule_texts']
            return f'subrreddit: r/{subrredit}. rule_text: {rule_text}. comment: {comment}.'

        def augment_context(row):
            context = row['context']
            subrredit = row['subreddit']
            rule_text = row['rule_texts']
            return [f'subrreddit: r/{subrredit}. rule_text: {rule_text}. comment: {comment}.' for comment in context]

        subreddits = df['subreddit'].tolist()
        df['final_comment'] = df['final_comment'].parallel_apply(reddit_clean)
        df['context'] = df['context'].parallel_apply(reddit_batch_clean)
        comments = df.parallel_apply(augment_comment, axis=1)
        contexts = df.parallel_apply(augment_context, axis=1)
        conversations = [x + [y] for x, y in zip(contexts, comments)]
        # rule_texts = df['rule_texts'].tolist()
        conv_lens = pd.Series(conversations).apply(len).tolist()

        conversations_1d = list(itertools.chain(*conversations))

        if model_name == 'bert-base-uncased':
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        elif model_name == 'gpt2':
            from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        else:  # t5-base
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("t5-base")

        # encode the conversations
        print('Tokenizing....')
        encodings = tokenizer(conversations_1d, padding='max_length', truncation=True, max_length=max_n_tokens,
                              return_tensors='pt')
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        print('Done\n')

        def slice_list(lst, chunk_sizes):
            result = []
            i = 0
            for size in chunk_sizes:
                result.append(lst[i:i + size])
                i += size
            return result

        input_ids = slice_list(input_ids, conv_lens)
        attention_mask = slice_list(attention_mask, conv_lens)

        # pad the conversations
        if model_name == 'bert-base-uncased':
            dummpy_input_ids = torch.tensor(
                [tokenizer.cls_token_id, tokenizer.sep_token_id] + [tokenizer.pad_token_id] * (max_n_tokens - 2))
        elif model_name == 'gpt2':
            dummpy_input_ids = torch.tensor(
                [tokenizer.bos_token_id, tokenizer.eos_token_id] + [tokenizer.pad_token_id] * (max_n_tokens - 2))
        else:
            dummpy_input_ids = torch.tensor(
                [tokenizer.pad_token_id, tokenizer.eos_token_id] + [tokenizer.pad_token_id] * (max_n_tokens - 2))
        dummpy_attention_mask = torch.tensor([1, 1] + [0] * (max_n_tokens - 2))

        def pad_conv(i):
            conv_len = conv_lens[i]
            n_padding = max_context_size + 1 - conv_len
            input_ids[i] = torch.cat([input_ids[i], dummpy_input_ids.repeat(n_padding, 1)], dim=0)
            attention_mask[i] = torch.cat([attention_mask[i], dummpy_attention_mask.repeat(n_padding, 1)], dim=0)

        indices = pd.Series(range(n))
        indices.apply(pad_conv)

        self.n = n
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.subreddits = subreddits
        self.conv_lens = conv_lens

    def __getitem__(self, index):
        item = {
            'input_ids': self.input_ids[index],
            'attention_mask': self.attention_mask[index],
            'subreddit': self.subreddits[index],
            'conv_len': self.conv_lens[index],
        }
        return item

    def __len__(self):
        return self.n


def create_normvio_prompt_dataset(conversations, subreddits, rules, max_context_size=5):
    from pandarallel import pandarallel
    pandarallel.initialize(nb_workers=4)

    last_comments = [conv[-1] for conv in conversations]
    contexts = [conv[:-1] for conv in conversations]
    df = pd.DataFrame({
        'final_comment': last_comments,
        'context': contexts,
        'subreddit': subreddits,
        'rule_texts': rules
    })

    def truncate_context(x):
        n = len(x)
        if n >= max_context_size:
            x = x[-max_context_size:]
        else:
            x = ['None.'] * (max_context_size - n) + x
        return x

    df['context'] = df['context'].parallel_apply(truncate_context)

    def reddit_clean(x):
        return p.tokenize(redditcleaner.clean(x))

    def reddit_batch_clean(x):
        return [p.tokenize(redditcleaner.clean(e)) for e in x]

    df['final_comment'] = df['final_comment'].parallel_apply(reddit_clean)
    df['context'] = df['context'].parallel_apply(reddit_batch_clean)

    from openprompt.data_utils import InputExample
    def create_input_example(row):
        meta = {
            'subreddit': row['subreddit'],
            'rule': row['rule_texts'],
        }
        for i in range(max_context_size):
            meta[f'comment{i}'] = row['context'][i]
        meta[f'comment{max_context_size}'] = row['final_comment']
        return InputExample(meta=meta)

    data = df.apply(create_input_example, axis=1).tolist()
    for i, each in enumerate(data):
        each.guid = i

    return data


def data_loader(conversations, subreddits, rules, batch_size, model_name='bert-base-uncased',
                max_context_size=5, max_n_tokens=128, n_workers=4):
    dataset = NormVioSeqInference(conversations, subreddits, rules, model_name=model_name,
                                  max_context_size=max_context_size, max_n_tokens=max_n_tokens, n_workers=n_workers)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers)
    return loader

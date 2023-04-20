import pandas as pd
import torch
import itertools
from torch.utils.data import Dataset, DataLoader
import redditcleaner
from multiprocessing import Process
import preprocessor as p
p.set_options(p.OPT.URL, p.OPT.EMOJI)

all_cats = ['incivility', 'harassment', 'spam', 'format', 'content',
            'off-topic', 'hatespeech', 'trolling', 'meta-rules']


class NormVioSeq(Dataset):
    def __init__(self, phase, max_context_size=5, max_n_tokens=128):
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=4)

        df = pd.read_csv(f'data/{phase}.csv', converters={'context': eval})
        n = df.shape[0]
        print(f'**********{phase} set, {n} comments**********')

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

        def reddit_batch_clean(x):
            x = x[:max_context_size]
            return [p.tokenize(redditcleaner.clean(e)) for e in x]

        def reddit_clean(x):
            return p.tokenize(redditcleaner.clean(x))

        subreddits = df['subreddit'].tolist()
        comments = df.parallel_apply(augment_comment, axis=1)
        contexts = df.parallel_apply(augment_context, axis=1)
        comments = comments.parallel_apply(reddit_clean).tolist()
        contexts = contexts.parallel_apply(reddit_batch_clean).tolist()
        conversations = [x+[y] for x, y in zip(contexts, comments)]
        # rule_texts = df['rule_texts'].tolist()
        for cat in all_cats:
            n_cat = df['cats'].apply(lambda x: cat in x).sum()
            print(f'{cat}: {n_cat/n:.2f}')
        print()
        cats = df['cats'].tolist()
        labels = df['bool_derail'].astype(int).tolist()
        conv_lens = pd.Series(conversations).apply(len).tolist()


        conversations_1d = list(itertools.chain(*conversations))

        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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
        dummpy_input_ids = torch.tensor([tokenizer.cls_token_id, tokenizer.sep_token_id] + [0]*(max_n_tokens-2))
        dummpy_attention_mask = torch.tensor([1, 1] + [0]*(max_n_tokens-2))

        def pad_conv(i):
            conv_len = conv_lens[i]
            n_padding = max_context_size + 1 - conv_len
            input_ids[i] = torch.cat([input_ids[i], dummpy_input_ids.repeat(n_padding, 1)], dim=0)
            attention_mask[i] = torch.cat([attention_mask[i], dummpy_attention_mask.repeat(n_padding, 1)], dim=0)

        indices = pd.Series(range(n))
        indices.parallel_apply(pad_conv)

        # n_workers = 4
        # for i in range(0, n, n_workers):
        #     processes = [Process(target=pad_conv, args=(j,)) for j in range(i, i + n_workers)]
        #     # start all processes
        #     for process in processes:
        #         process.start()
        #     # wait for all processes to complete
        #     for process in processes:
        #         process.join()

        # for i in range(n):
        #     conv_len = conv_lens[i]
        #     n_padding = max_context_size + 1 - conv_len
        #     input_ids[i] = torch.cat([input_ids[i], dummpy_input_ids.repeat(n_padding, 1)], dim=0)
        #     attention_mask[i] = torch.cat([attention_mask[i], dummpy_attention_mask.repeat(n_padding, 1)], dim=0)

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.subreddits = subreddits
        self.conv_lens = conv_lens
        self.cats = cats
        self.labels = labels

    def __getitem__(self, index):
        item = {
            'input_ids': self.input_ids[index],
            'attention_mask': self.attention_mask[index],
            'subreddit': self.subreddits[index],
            'conv_len': self.conv_lens[index],
            'cat': self.cats[index],
            'label': self.labels[index]
        }
        return item

    def __len__(self):
        return len(self.labels)


class NormVioSeqInference(Dataset):
    def __init__(self, conversations, subreddits, rules, max_context_size=5, max_n_tokens=128):
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=4)

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

        def reddit_batch_clean(x):
            x = x[:max_context_size]
            return [p.tokenize(redditcleaner.clean(e)) for e in x]

        def reddit_clean(x):
            return p.tokenize(redditcleaner.clean(x))

        comments = [conv[-1] for conv in conversations]
        contexts = [conv[:-1] for conv in conversations]

        df = pd.DataFrame({'context': contexts,
                           'final_comment': comments,
                           'subreddit': subreddits,
                           'rule_texts': rules})
        n = df.shape[0]

        comments = df.parallel_apply(augment_comment, axis=1)
        contexts = df.parallel_apply(augment_context, axis=1)
        comments = comments.parallel_apply(reddit_clean).tolist()
        contexts = contexts.parallel_apply(reddit_batch_clean).tolist()
        conversations = [x+[y] for x, y in zip(contexts, comments)]

        conv_lens = pd.Series(conversations).apply(len).tolist()
        conversations_1d = list(itertools.chain(*conversations))

        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # encode the conversations
        encodings = tokenizer(conversations_1d, padding='max_length', truncation=True, max_length=max_n_tokens,
                              return_tensors='pt')
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']

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
        dummpy_input_ids = torch.tensor([tokenizer.cls_token_id, tokenizer.sep_token_id] + [0]*(max_n_tokens-2))
        dummpy_attention_mask = torch.tensor([1, 1] + [0]*(max_n_tokens-2))
        for i in range(n):
            conv_len = conv_lens[i]
            n_padding = max_context_size + 1 - conv_len
            input_ids[i] = torch.cat([input_ids[i], dummpy_input_ids.repeat(n_padding, 1)], dim=0)
            attention_mask[i] = torch.cat([attention_mask[i], dummpy_attention_mask.repeat(n_padding, 1)], dim=0)

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.subreddits = subreddits
        self.conv_lens = conv_lens
        self.n = n

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


def data_loader(model_name, phase, batch_size, max_context_size=5, max_n_tokens=128, n_workers=4):
    shuffle = True if phase == 'train' else False
    dataset = NormVioSeq(phase=phase, max_context_size=max_context_size, max_n_tokens=max_n_tokens)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers)
    return loader


def data_loader_inference(model_name, conversations, subreddits, rules, batch_size,
                          max_context_size=5, max_n_tokens=128, n_workers=4):
    dataset = NormVioSeqInference(conversations, subreddits, rules, max_context_size, max_n_tokens)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers)
    return loader

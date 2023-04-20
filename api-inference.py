from flask import Flask, jsonify, request
from flask_restful import Resource, Api
import argparse
import json
from evaluator import Evaluator


class NormVioDetect(Resource):
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def post(self):
        '''

        :param convs: the list of conversations, separated by aaaaa. Within each conversation, each utterance is separated by bbbbb
        :param reddits: the names of subreddits, separated by aaaaa.
        :param rules: the list of rules aaaaa.
        :return: a list of probabilities. json format
        '''

        data = request.get_json(force=True)
        convs = data['convs']
        reddits = data['reddits']
        rules = data['rules']
        probs = self.evaluator.inference(convs, reddits, rules)
        data['probs'] = probs

        return jsonify(data)


# driver function
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

    prefix = f'results/{args.model_name}/{args.idx}'
    with open(f'{prefix}/config.json', 'r') as f:
        config = json.load(f)
    for name in ['max_context_size', 'max_n_tokens', 'n_rnn_layers', 'dropout']:
        args.__setattr__(name, config[name])

    print(json.dumps(args.__dict__, indent=2))

    engine = Evaluator(args)

    # creating the flask app
    app = Flask(__name__)
    # creating an API object
    api = Api(app)
    api.add_resource(NormVioDetect, '/api', resource_class_kwargs={'evaluator': engine})

    app.run(debug=True, port=5000)
import argparse
import csv
import numpy as np
from tqdm import tqdm

from allennlp.common.params import Params
from allennlp.data.dataset_readers.snli import SnliReader
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.models.model import Model
from allennlp.predictors.predictor import Predictor


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate dataset accuracy on a variety of NLI models.')
    parser.add_argument('--model_params', type=str,
                        help='the .jsonnet model configuration', required=True)
    parser.add_argument('--model_dir', type=str,
                        help='the model serialization directory', required=True)
    parser.add_argument('--eval_set', type=str,
                        help='the evaluation data set', required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    tokenizer = WordTokenizer()
    token_indexer = SingleIdTokenIndexer(lowercase_tokens=True)
    reader = SnliReader(tokenizer=tokenizer, token_indexers={'tokens': token_indexer})

    print('Loading model params from ' + args.model_params)
    model_params = Params.from_file(args.model_params)

    print('Loading model from ' + args.model_dir)
    model = Model.load(model_params, args.model_dir)

    print('Evaluating model over ' + args.eval_set)
    with open(args.eval_set, 'r') as f:
        entailed_correct = 0
        entailed_incorrect = 0
        nonentailed_correct = 0
        nonentailed_incorrect = 0
        total_correct = 0
        total_incorrect = 0
        errors_from_model = 0

        tsv_reader = csv.DictReader(f, delimiter='\t')
        for row in tqdm(tsv_reader):
            premise = row['sentence1']
            hypothesis = row['sentence2']

            instance = reader.text_to_instance(premise, hypothesis)
            try:
                logits = model.forward_on_instance(instance)['label_logits']
            except:
                errors_from_model += 1
                continue
            label_id = np.argmax(logits)
            label = model.vocab.get_token_from_index(label_id, 'labels')

            # Contradiction and neutral are both "non-entailment".
            is_pred_entailment = label == 'entailment'
            is_gold_entailment = row['gold_label'] == 'entailment'
            is_correct = is_pred_entailment == is_gold_entailment

            if is_correct and is_gold_entailment:
                entailed_correct += 1
                total_correct += 1
            if is_correct and (not is_gold_entailment):
                nonentailed_correct += 1
                total_correct += 1
            if (not is_correct) and is_gold_entailment:
                entailed_incorrect += 1
                total_incorrect += 1
            if (not is_correct) and (not is_gold_entailment):
                nonentailed_incorrect += 1
                total_incorrect += 1

    if total_correct + total_incorrect == 0:
        total_accuracy = 0.0
    else:
        total_accuracy = total_correct * 1.0 / (total_correct + total_incorrect)

    if entailed_correct + entailed_incorrect == 0:
        entailed_accuracy = 0.0
    else:
        entailed_accuracy = entailed_correct * 1.0 / (entailed_correct + entailed_incorrect)

    if nonentailed_correct + nonentailed_incorrect == 0:
        nonentailed_accuracy = 0.0
    else:
        nonentailed_accuracy = nonentailed_correct * 1.0 / (nonentailed_correct + nonentailed_incorrect)

    print('Total correct: ' + str(total_correct))
    print('    Entailed: ' + str(entailed_correct))
    print('    Non-entailed: ' + str(nonentailed_correct))
    print('Total incorrect: ' + str(total_incorrect))
    print('    Entailed: ' + str(entailed_incorrect))
    print('    Non-entailed: ' + str(nonentailed_incorrect))
    print('Total accuracy: ' + str(total_accuracy))
    print('    Entailed: ' + str(entailed_accuracy))
    print('    Non-entailed: ' + str(nonentailed_accuracy))
    if (errors_from_model > 0):
        print("Errors from model: " + str(errors_from_model))


if __name__ == '__main__':
    main()

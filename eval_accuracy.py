import csv
import numpy as np
from tqdm import tqdm

from allennlp.common.params import Params
from allennlp.data.dataset_readers.snli import SnliReader
from allennlp.data.token_indexers.single_id_token_indexer import SingleIdTokenIndexer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.models.model import Model
from allennlp.predictors.predictor import Predictor

MODEL_PARAMS_FILE = 'model_params/da.jsonnet'
MODEL_DIR = 'models/da'
INPUT_TSV = 'eval_sets/hans.tsv'

tokenizer = WordTokenizer()
token_indexer = SingleIdTokenIndexer(lowercase_tokens=True)
reader = SnliReader(tokenizer=tokenizer, token_indexers={'tokens': token_indexer})

print('Loading model params from ' + MODEL_PARAMS_FILE)
model_params = Params.from_file(MODEL_PARAMS_FILE)

print('Loading model from ' + MODEL_DIR)
model = Model.load(model_params, MODEL_DIR)

print('Evaluating model over ' + INPUT_TSV)
with open(INPUT_TSV, 'r') as f:
    entailed_correct = 0
    entailed_incorrect = 0
    nonentailed_correct = 0
    nonentailed_incorrect = 0
    total_correct = 0
    total_incorrect = 0

    tsv_reader = csv.DictReader(f, delimiter='\t')
    for row in tqdm(tsv_reader):
        premise = row['sentence1']
        hypothesis = row['sentence2']

        instance = reader.text_to_instance(premise, hypothesis)
        logits = model.forward_on_instance(instance)['label_logits']
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

    print('Total correct: ' + str(total_correct))
    print('    Entailed: ' + str(entailed_correct))
    print('    Non-entailed: ' + str(nonentailed_correct))
    print('Total incorrect: ' + str(total_incorrect))
    print('    Entailed: ' + str(entailed_incorrect))
    print('    Non-entailed: ' + str(nonentailed_incorrect))
    print('Total accuracy: ' + str(total_correct * 1.0 / (total_correct + total_incorrect)))
    print('    Entailed: ' + str(entailed_correct * 1.0 / (entailed_correct + entailed_incorrect)))
    print('    Non-entailed: ' + str(nonentailed_correct * 1.0 / (nonentailed_correct + nonentailed_incorrect)))

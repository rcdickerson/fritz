import argparse
import csv
from nltk import download as nltk_download
from nltk import Tree
from nltk.corpus import wordnet
import random
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Perturb SNLI / MNLI style data')
    parser.add_argument('infile', type=str,
                        help='file path of the input data to manipulate')
    parser.add_argument('-o', '--output', type=str, default='out.tsv',
                        help='file path to output perturbed data')
    parser.add_argument('-t', '--type', type=str, dest='ptype', default='shuffle',
                        choices=['shuffle', 'neghyp', 'hverbsyn', 'hverbant',
                                 'hadvsyn', 'hadvant', 'hnounsyn', 'hnounant'],
                        help='the type of perturbation to perform, default is swap')
    return parser.parse_args()


def rand_syn(word, pos):
    syns = [lemma.name()
            for syn in wordnet.synsets(word, pos=pos)
            for lemma in syn.lemmas()]
    return random.choice(syns).replace('_', ' ') if len(syns) > 0 else word

def rand_ant(word, pos):
    ants = [ant.name()
            for syn in wordnet.synsets(word, pos)
            for lemma in syn.lemmas()
            for ant in lemma.antonyms()]
    return random.choice(ants).replace('_', ' ') if len(ants) > 0 else word

def transform_sentence(parse, pos_prefix, transform_func):
    transform = []
    any_replacements = False
    for (word, pos) in parse:
        if pos.startswith(pos_prefix):
            repl_word = transform_func(word)
            if word != repl_word: any_replacements = True
            transform.append(repl_word)
        else:
            transform.append(word)
    return (any_replacements, fix_contractions(' '.join(transform)))

# The MNLI parse breaks contractions up as separate words,
# this is a helper method to put them back together.
def fix_contractions(sentence):
    return sentence.replace(" '", "'").replace(" n'", "n'")

def perturb(ptype, row):
    premise = row['sentence1']
    hypothesis = row['sentence2']
    is_entailment = row['gold_label'] == 'entailment'
    is_neutral = row['gold_label'] == 'neutral'
    perturbed = False

    if ptype == 'shuffle':
        (pwords, hwords) = (premise.split(), hypothesis.split())
        random.shuffle(pwords)
        random.shuffle(hwords)
        premise = ' '.join(pwords)
        hypothesis = ' '.join(hwords)
        is_entailment = False
        perturbed = True

    elif ptype == 'neghyp':
        hypothesis = 'It is not the case that ' + hypothesis[0].lower() + hypothesis[1:]
        is_entailment = False if is_neutral else not is_entailment
        perturbed = True

    elif ptype == 'hverbsyn':
        hparse = Tree.fromstring(row['sentence2_parse']).pos()
        (perturbed, hypothesis) = transform_sentence(hparse, 'VB', lambda word: rand_syn(word, 'v'))

    elif ptype == 'hverbant':
        hparse = Tree.fromstring(row['sentence2_parse']).pos()
        (perturbed, hypothesis) = transform_sentence(hparse, 'VB', lambda word: rand_ant(word, 'v'))
        if perturbed:
            is_entailment = False if is_neutral else not is_entailment

    elif ptype == 'hadvsyn':
        hparse = Tree.fromstring(row['sentence2_parse']).pos()
        (perturbed, hypothesis) = transform_sentence(hparse, 'RB', lambda word: rand_syn(word, 'r'))
    elif ptype == 'hadvant':
        hparse = Tree.fromstring(row['sentence2_parse']).pos()
        (perturbed, hypothesis) = transform_sentence(hparse, 'RB', lambda word: rand_ant(word, 'r'))
        if perturbed:
            is_entailment = False if is_neutral else not is_entailment

    elif ptype == 'hnounsyn':
        hparse = Tree.fromstring(row['sentence2_parse']).pos()
        (perturbed, hypothesis) = transform_sentence(hparse, 'NN', lambda word: rand_syn(word, 'n'))
    elif ptype == 'hnounant':
        hparse = Tree.fromstring(row['sentence2_parse']).pos()
        (perturbed, hypothesis) = transform_sentence(hparse, 'NN', lambda word: rand_ant(word, 'n'))
        if perturbed:
            is_entailment = False if is_neutral else not is_entailment


    return (perturbed,
            {'gold_label': 'entailment' if is_entailment else 'non-entailment',
            'sentence1' : premise,
            'sentence2' : hypothesis})


def main():
    args = parse_args()
    nltk_download('wordnet')

    out_fieldnames = ['gold_label', 'sentence1', 'sentence2']

    with open(args.infile, 'r') as infile, open(args.output, 'w') as outfile:
        tsv_reader = csv.DictReader(infile, delimiter='\t')
        tsv_writer = csv.DictWriter(outfile, fieldnames=out_fieldnames, delimiter='\t')
        tsv_writer.writeheader()
        for row in tqdm(tsv_reader):
            (perturbed, perturbation) = perturb(args.ptype, row)
            if perturbed: tsv_writer.writerow(perturbation)


if __name__ == '__main__':
    main()

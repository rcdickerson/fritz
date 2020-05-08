import argparse
import csv
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
                        choices=['shuffle', 'neghyp'],
                        help='the type of perturbation to perform, default is swap')
    return parser.parse_args()


def perturb(ptype, row):
    premise = row['sentence1']
    hypothesis = row['sentence2']
    is_entailment = row['gold_label'] == 'entailment'
    is_neutral = row['gold_label'] == 'neutral'

    (pwords, hwords) = (premise.split(), hypothesis.split())

    if ptype == 'shuffle':
        random.shuffle(pwords)
        random.shuffle(hwords)
        premise = ' '.join(pwords)
        hypothesis = ' '.join(hwords)
        is_entailment = False
    elif ptype == 'neghyp':
        hypothesis = 'It is not the case that ' + hypothesis[0].lower() + hypothesis[1:]
        is_entailment = False if is_neutral else not is_entailment

    return {'gold_label': 'entailment' if is_entailment else 'non-entailment',
            'sentence1' : premise,
            'sentence2' : hypothesis}


def main():
    args = parse_args()

    out_fieldnames = ['gold_label', 'sentence1', 'sentence2']

    with open(args.infile, 'r') as infile, open(args.output, 'w') as outfile:
        tsv_reader = csv.DictReader(infile, delimiter='\t')
        tsv_writer = csv.DictWriter(outfile, fieldnames=out_fieldnames, delimiter='\t')
        tsv_writer.writeheader()
        for row in tqdm(tsv_reader):
            tsv_writer.writerow(perturb(args.ptype, row))


if __name__ == '__main__':
    main()

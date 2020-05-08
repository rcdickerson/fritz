import argparse
import csv
import random
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description='Counts the number of each gold label in a dataset')
    parser.add_argument('infile', type=str,
                        help='file path of the dataset to count')
    return parser.parse_args()

def main():
    args = parse_args()
    counts = {}

    with open(args.infile, 'r') as infile:
        reader = csv.DictReader(infile, delimiter='\t')
        for row in tqdm(reader):
            label = row['gold_label']
            if label in counts:
                counts[label] += 1
            else:
                counts[label] = 1

    for label in counts:
        print(label + ': ' + str(counts[label]))


if __name__ == '__main__':
    main()

import os
import argparse
from typing import List

from viterbi import viterbi


def calculate_accuracy(output_list: str, target_list: str):
    total_words = 0
    correct_words = 0
    total_sentences = 0
    correct_sentences = 0

    with open(output_list, 'r', encoding='utf-8') as output, \
            open(target_list, 'r', encoding='utf-8') as target:
        for output_line, target_line in zip(output, target):
            total_sentences += 1
            if output_line == target_line:
                correct_sentences += 1

            total_words += len(target_line)
            for output_word, target_word in zip(output_line, target_line):
                if output_word == target_word:
                    correct_words += 1

    word_accuracy = correct_words / total_words * 100 if total_words else 0
    sentence_accuracy = correct_sentences / \
        total_sentences * 100 if total_sentences else 0

    print(f'Word Accuracy: {word_accuracy:.3f}%')
    print(f'Sentence Accuracy: {sentence_accuracy:.3f}%')


def get_args():
    parser = argparse.ArgumentParser(description='Pinyin Input Method')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='location of the input pinyin file')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='location of the output file')
    parser.add_argument('-a', '--answer', type=str,
                        default='../data/std_output.txt', help='location of the answer file')
    parser.add_argument('-s', '--smooth', type=List[float],
                        default=[0.9, 0.7], help='smoothing parameters')
    parser.add_argument('-n', '--ngram', type=int, default=3,
                        help='ngram model to use (1, 2, or 3(recommended)')

    args = parser.parse_args()
    if not os.path.exists(args.input):
        parser.error('The input file does not exist!')
    return args


if __name__ == '__main__':
    args = get_args()

    output_list = []
    with open(args.input, 'r', encoding='utf-8') as file:
        for line in file:
            output_list.append(viterbi(line, args.smooth, args.ngram))

    with open(args.output, 'w', encoding='utf-8') as file:
        for line in output_list:
            file.write(line + '\n')

    calculate_accuracy(args.output, args.answer)

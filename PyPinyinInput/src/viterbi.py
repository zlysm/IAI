import os
import math
import pickle
from typing import List


# model_path = './models'
model_path = './models/all_corpus'
# model_path = './models/sina_and_smp'
# model_path = './models/sina'

unigram_path = os.path.join(model_path, 'unigrams.pkl')
bigram_path = os.path.join(model_path, 'bigrams.pkl')
trigram_path = os.path.join(model_path, 'trigrams.pkl')
dictionary_path = os.path.join(model_path, 'dictionary.pkl')

if not os.path.exists(model_path):
    print('Error! Please run train.py first!')
    exit()

unigrams = pickle.load(open(unigram_path, 'rb'))
bigrams = pickle.load(open(bigram_path, 'rb'))
trigrams = pickle.load(open(trigram_path, 'rb'))
dictionary = pickle.load(open(dictionary_path, 'rb'))

uni_sum = sum(unigrams.values())


def cost(words: str, smooth: List[float]) -> float:
    """
    Calculate the cost of a word sequence.

    Parameters:
        words (str): The word sequence.
        smooth (List[float]): Smoothing parameters for bigram and trigram.

    Returns:
        float: The cost of the word sequence.
    """
    [a, b] = smooth  # smoothing parameters for bigram and trigram
    ngram = len(words)
    eps = 1e-7
    if ngram == 1:
        return -math.log(unigrams.get(words, 0) / uni_sum or eps)
    elif ngram == 2:
        return -math.log(a * bigrams.get(words, 0) / unigrams.get(words[0], 1)
                         + (1 - a) * unigrams.get(words[1], 0) / uni_sum or eps)
    else:
        return -math.log(b * trigrams.get(words, 0) / bigrams.get(words[:2], 1)
                         + (1 - b) * (a * bigrams.get(words[1:], 0) / unigrams.get(words[1], 1)
                                      + (1 - a) * unigrams.get(words[2], 0) / uni_sum) or eps)


def viterbi(sentence: str, smooth: List[float], ngram: int) -> str:
    """
    Apply the Viterbi algorithm to find the most likely word sequence from a pinyin sequence.

    Parameters:
        sentence (str): The pinyin sequence, separated by spaces.
        smooth (List[float]): Smoothing parameters for bigram and trigram.
        ngram (int): The ngram model to use (1, 2, or 3).

    Returns:
        str: The most likely word sequence.
    """
    pinyins = sentence.split()
    try:
        viterbi_graph = [dictionary[pinyin] for pinyin in pinyins]
    except:
        return 'Error! Please check the pinyin sequence!'

    words_list = [(word, cost(word, smooth)) for word in viterbi_graph[0]]

    for i in range(1, len(viterbi_graph)):
        words = []
        for cur in viterbi_graph[i]:
            min_cost = float('inf')
            min_words = ''
            for prev_word, prev_cost in words_list:
                if ngram == 1:
                    c = cost(cur, smooth) + prev_cost
                elif ngram == 2 or (ngram == 3 and i == 1):
                    c = cost(prev_word[-1] + cur, smooth) + prev_cost
                else:
                    c = cost(prev_word[-2:] + cur, smooth) + prev_cost

                if c < min_cost:
                    min_cost = c
                    min_words = prev_word + cur
            words.append((min_words, min_cost))
        words_list = words

    return min(words_list, key=lambda x: x[1])[0]


if __name__ == '__main__':
    while (True):
        sentence = input('Please input the pinyin sentence:\n')
        print(viterbi(sentence, [0.9, 0.7], 3))

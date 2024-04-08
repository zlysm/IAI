import math
import json
import sys

dictionary = {}  # key: pinyin, value: list of words
with open('./word2pinyin.txt', 'r', encoding='utf-8') as file:
    for line in file:
        value, key = line.split()
        if key in dictionary:
            dictionary[key].append(value)
        else:
            dictionary[key] = [value]

unigrams = {}
uni_sum = 0
with open('./1_word.txt', 'r', encoding='utf-8') as file:
    data = json.load(file)
    for d in data:
        for key, value in zip(data[d]['words'], data[d]['counts']):
            unigrams[key] = unigrams.get(key, 0) + value
            uni_sum += value

bigrams = {}
with open('./2_word.txt', 'r', encoding='utf-8') as file:
    data = json.load(file)
    for d in data:
        for key, value in zip(data[d]['words'], data[d]['counts']):
            bigrams[key] = bigrams.get(key, 0) + value


def cost(words: str, a: float) -> float:
    model = len(words)
    if model == 1:
        return -math.log(unigrams.get(words, 0) / uni_sum or 1e-7)
    else:
        word1, word2 = words.split()
        return -math.log(a * bigrams.get(words, 0) / unigrams.get(word1, 1)
                         + (1 - a) * unigrams.get(word2, 0) / uni_sum or 1e-7)


def viterbi(sentence: str, a: float) -> str:
    pinyins = sentence.split()

    viterbi_graph = [dictionary[pinyin] for pinyin in pinyins]
    words_list = [(word, cost(word, a)) for word in viterbi_graph[0]]

    for i in range(1, len(viterbi_graph)):
        words = []
        for cur in viterbi_graph[i]:
            min_cost = float('inf')
            min_words = ''
            for prev_word, prev_cost in words_list:
                c = cost(prev_word[-1] + ' ' + cur, a) + prev_cost

                if c < min_cost:
                    min_cost = c
                    min_words = prev_word + cur
            words.append((min_words, min_cost))
        words_list = words

    return min(words_list, key=lambda x: x[1])[0]


if __name__ == '__main__':
    while (True):
        line = sys.stdin.readline().strip()
        if not line:
            break
        print(viterbi(line, 0.9555))

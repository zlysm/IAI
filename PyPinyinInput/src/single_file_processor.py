import re
import json
from tqdm import tqdm


class SingleFileProcessor:
    def __init__(self, file_path, chinese_characters):
        """
        Initialize SingleFileProcessor object.

        Parameters:
            file_path (str): Path to the file to process.
            chinese_characters (set): Set containing all Chinese characters.
        """
        self.file_path = file_path
        self.chinese_characters = chinese_characters
        self.pattern = re.compile(r'[\u4e00-\u9fff]+')
        self.unigrams = {}
        self.bigrams = {}
        self.trigrams = {}

    def generate_legal_words(self):
        """
        Process the corpus file and extract Chinese texts from it.
        """
        words = []  # words = [sentence1, sentence2, ...]
        with open(self.file_path, 'r', encoding='gbk' if self.file_path.endswith('.txt') else 'utf-8') as file:
            for line in file:
                try:
                    sentence = json.loads(line)
                    if 'html' in sentence:  # sina_news_gbk
                        words.extend(self.pattern.findall(sentence['html']))
                        words.extend(self.pattern.findall(sentence['title']))
                    elif 'desc' in sentence:  # baike2018qa
                        words.extend(self.pattern.findall(sentence['desc']))
                        words.extend(self.pattern.findall(sentence['answer']))
                except:  # SMP2020
                    words.extend(self.pattern.findall(line))
        self.words = words  # list of sentences

    def count_ngram(self):
        """
        Count unigram, bigram, and trigram word frequency models.
        """
        for words in tqdm(self.words):
            for i in range(len(words)):
                # Unigram
                if not words[i] in self.chinese_characters:
                    continue
                self.unigrams[words[i]] = self.unigrams.get(words[i], 0) + 1

                # Bigram
                if i >= len(words) - 1 or words[i + 1] not in self.chinese_characters:
                    continue
                self.bigrams[words[i: i + 2]] = self.bigrams.get(
                    words[i: i + 2], 0) + 1

                # Trigram
                if i >= len(words) - 2 or words[i + 2] not in self.chinese_characters:
                    continue
                self.trigrams[words[i: i + 3]] = self.trigrams.get(
                    words[i: i + 3], 0) + 1

    def train(self):
        """
        Train the model by generating legal texts and counting n-grams.
        """
        self.generate_legal_words()
        self.count_ngram()

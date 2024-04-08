import os
import pickle
from tqdm import tqdm
from multiprocessing import Pool

from single_file_processor import SingleFileProcessor


class ChineseTextProcessor:
    """
    Processing text corpus and training Ngram models.
    """

    def __init__(self, hanzi_table_path: str, corpus_folder_path: str):
        """
        Initialize ChineseTextProcessor object.

        Parameters:
            hanzi_table_path (str): Path to the Chinese character table file.
            corpus_folder_path (str): Path to the corpus folder.
        """
        self.chinese_characters = self.load_chinese_characters(
            hanzi_table_path)
        self.corpus_folder = corpus_folder_path
        self.unigrams = {}
        self.bigrams = {}
        self.trigrams = {}

    def load_chinese_characters(self, file_path: str) -> set:
        """
        Load the Chinese character table.

        Parameters:
            file_path (str): Path to the Chinese character table file.

        Returns:
            set: A set containing all Chinese characters.
        """
        with open(file_path, 'r', encoding='gbk') as file:
            chinese_characters = file.read().strip()
        return set(chinese_characters)

    def process_single_file(self, file_path: str) -> tuple[dict, dict, dict]:
        """ 
        Process a single file, including training Ngram models.

        Parameters:
            file_path (str): Path to the file to process.

        Returns:
            tuple: Contains the unigrams, bigrams, and trigrams dictionaries of the single file.
        """
        single_processor = SingleFileProcessor(
            file_path, self.chinese_characters)
        single_processor.train()
        return single_processor.unigrams, single_processor.bigrams, single_processor.trigrams

    def process_all_corpus(self):
        """
        Process all files in the corpus folder and update Ngram models.
        """
        files = []
        for root, _, file_names in os.walk(self.corpus_folder):
            for file_name in file_names:
                files.append(os.path.join(root, file_name))

        with Pool() as pool:
            results = list(tqdm(pool.imap(self.process_single_file, files), total=len(
                files), desc='Processing Corpus'))

        # Merge results into Ngram models
        for unigrams, bigrams, trigrams in tqdm(results, desc='Merging Results'):
            for k, v in unigrams.items():
                self.unigrams[k] = self.unigrams.get(k, 0) + v
            for k, v in bigrams.items():
                self.bigrams[k] = self.bigrams.get(k, 0) + v
            for k, v in trigrams.items():
                self.trigrams[k] = self.trigrams.get(k, 0) + v

        # Filter out low frequency words
        min_freq = 5
        self.unigrams = {k: v for k, v in self.unigrams.items()
                         if v > min_freq}
        self.bigrams = {k: v for k, v in self.bigrams.items()
                        if v > min_freq}
        self.trigrams = {k: v for k, v in self.trigrams.items()
                         if v > min_freq}


def train_ngram():
    """
    Train Ngram models and save them.
    """
    hanzi_table_path = './PinyinCharMap/一二级汉字表.txt'
    corpus_folder_path = './corpus'

    # Create a ChineseTextProcessor instance
    processor = ChineseTextProcessor(hanzi_table_path, corpus_folder_path)
    processor.process_all_corpus()

    # Save Ngram models
    unigrams_path = './models/unigrams.pkl'
    bigrams_path = './models/bigrams.pkl'
    trigrams_path = './models/trigrams.pkl'
    with open(unigrams_path, 'wb') as file:
        pickle.dump(processor.unigrams, file)
    with open(bigrams_path, 'wb') as file:
        pickle.dump(processor.bigrams, file)
    with open(trigrams_path, 'wb') as file:
        pickle.dump(processor.trigrams, file)


def train_dictionary():
    """
    Train a dictionary and save it.
    """
    hanzi_table_path = './PinyinCharMap/一二级汉字表.txt'
    pinyin_table_path = './PinyinCharMap/拼音汉字表.txt'

    chinese_characters = open(hanzi_table_path, 'r',
                              encoding='gbk').read().strip()
    dictionary = {}
    with open(pinyin_table_path, 'r', encoding='gbk') as file:
        for line in file:
            dic = line.split()
            key, values = dic[0], dic[1:]
            new_values = [
                value for value in values if value in chinese_characters]
            dictionary[key] = new_values

    # Save the dictionary
    dictionary_path = './models/dictionary.pkl'
    with open(dictionary_path, 'wb') as file:
        pickle.dump(dictionary, file)


if __name__ == '__main__':
    # Create the 'models' folder if it doesn't exist
    if not os.path.exists('./models'):
        os.makedirs('./models')

    train_ngram()
    train_dictionary()

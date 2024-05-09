import os
from gensim.models import KeyedVectors
from matplotlib import pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader


class DatasetProcessor():
    def __init__(self, folder_path: str, max_length: int, batch_size: int):
        """
        Initialize the DatasetProcessor object

        Parameters:
            - folder_path (str): path to the folder containing the text files
            - max_length (int): maximum length of the text
            - batch_size (int): batch size for the DataLoader
        """
        self.folder_path = folder_path
        self.max_length = max_length
        self.batch_size = batch_size
        self.wordID = self.generate_wordID()

    def generate_wordID(self) -> dict:
        """
        Generate a dictionary mapping words to unique IDs

        Returns:
            - wordID (Dict[str, int]): key is the word, value is the unique ID
        """
        wordID = {}
        files = [os.path.join(self.folder_path, file) for file in os.listdir(
            self.folder_path) if file.endswith('.txt')]

        for file in files:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    texts = line.strip().split()
                    for word in texts[1:]:
                        if word not in wordID:
                            wordID[word] = len(wordID) + 1

        return wordID

    def generate_dataloader(self, file_path: str) -> DataLoader:
        """
        Generate a DataLoader for the given file, with the specified batch size.

        Parameters:
            - file_path (str): path to the text file

        Returns:
            - DataLoader: DataLoader object for the file
        """
        data = []
        labels = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                texts = line.strip().split()
                label = int(texts[0])
                # truncate the text if it exceeds max_length
                id_list = [self.wordID.get(word, 0)
                           for word in texts[1:]][:self.max_length]
                # pad the text if it is less than max_length
                id_list = id_list + [0] * \
                    max(0, self.max_length - len(id_list))
                data.append(id_list)
                labels.append(label)

        data = torch.tensor(data, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(data, labels)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def generate_word2vec(self, word2vec_path: str) -> torch.Tensor:
        """
        Generate an embedding matrix using the pre-trained Word2Vec model

        Parameters:
            - word2vec_path (str): path to the Word2Vec model

        Returns:
            - word2vec (torch.Tensor): embedding matrix for the words
        """
        model = KeyedVectors.load_word2vec_format(
            word2vec_path, binary=True)
        word2vec = torch.zeros(len(self.wordID) + 1, model.vector_size)
        for word, value in self.wordID.items():
            if word in model:
                word2vec[value] = torch.tensor(model[word])

        return word2vec

    def generate_all(self) -> tuple:
        """
        Generate the DataLoaders and the word2vec matrix

        Returns:
            - train_loader (DataLoader): DataLoader object for the training data
            - test_loader (DataLoader): DataLoader object for the testing data
            - val_loader (DataLoader): DataLoader object for the validation data
            - word2vec (torch.Tensor): embedding matrix for the words
        """
        loader_files = {
            'train_loader.pth': 'train.txt',
            'test_loader.pth': 'test.txt',
            'validation_loader.pth': 'validation.txt',
            'word2vec.pth': 'wiki_word2vec_50.bin'
        }

        data_loaders = {}
        for loader_name, file_name in loader_files.items():
            loader_path = os.path.join(self.folder_path, loader_name)
            file_path = os.path.join(self.folder_path, file_name)

            if os.path.exists(loader_path):
                data_loaders[loader_name] = torch.load(loader_path)
            else:
                if file_name.endswith('.txt'):
                    data_loaders[loader_name] = self.generate_dataloader(
                        file_path)
                elif file_name.endswith('.bin'):
                    data_loaders[loader_name] = self.generate_word2vec(
                        file_path)
                torch.save(data_loaders[loader_name], loader_path)

        return tuple(data_loaders.values())


def plot_metrics(metrics):
    metric_names = ['loss', 'accuracy', 'F_score']
    _, axs = plt.subplots(3, 3, figsize=(20, 15))

    for i, data_type in enumerate(['train', 'test', 'val']):
        for j, metric in enumerate(metric_names):
            data = [metrics[network][f'{data_type}_{metric}']
                    for network in metrics]
            for k, network in enumerate(metrics):
                axs[i, j].plot(data[k], label=f'{network}')
            axs[i, j].set_title(
                f'{data_type.capitalize()} {metric.capitalize()}')
            axs[i, j].set_xlabel('Epoch')
            axs[i, j].set_ylabel('Value')
            axs[i, j].legend()

    if not os.path.exists('figs'):
        os.makedirs('figs')
    plt.tight_layout()
    plt.savefig('figs/metrics.png')

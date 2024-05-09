import argparse
from tqdm import tqdm
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim

from utils import DatasetProcessor, plot_metrics
from networks import CNN, RNN, MLP


def get_args():
    parser = argparse.ArgumentParser(description='sentiment analysis')
    parser.add_argument('-n', '--network', type=str,
                        default='CNN', help='network type')
    parser.add_argument('-e', '--epochs', type=int,
                        default=10, help='number of epochs')
    parser.add_argument('-m', '--max_length', type=int,
                        default=50, help='max length of text')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=64, help='batch size')
    parser.add_argument('-lr', '--learning_rate', type=float,
                        default=0.001, help='learning rate')

    return parser.parse_args()


def get_metrics(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    correct_num = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for _, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item()
            correct_num += (outputs.argmax(dim=1) == labels).sum().item()
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    total_loss /= len(data_loader)
    accuracy = correct_num / len(data_loader.dataset)
    F_score = f1_score(all_labels, all_preds)

    return total_loss, accuracy, F_score


def train(model: nn.Module, args, train_loader, test_loader, val_loader):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5)

    metrics_data = {
        'train_loss': [],
        'train_accuracy': [],
        'train_F_score': [],
        'test_loss': [],
        'test_accuracy': [],
        'test_F_score': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_F_score': []
    }

    for _ in tqdm(range(args.epochs), desc=f'{model.__name__} training'):
        model.train()
        train_loss = 0
        correct_num = 0
        all_preds = []
        all_labels = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct_num += (outputs.argmax(dim=1) == labels).sum().item()

            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
        scheduler.step()

        train_loss /= len(train_loader)
        train_accuracy = correct_num / len(train_loader.dataset)
        train_F_score = f1_score(all_labels, all_preds)

        test_loss, test_accuracy, test_F_score = get_metrics(
            model, test_loader, criterion)

        val_loss, val_accuracy, val_F_score = get_metrics(
            model, val_loader, criterion)

        epoch_data = [
            train_loss, train_accuracy, train_F_score,
            test_loss, test_accuracy, test_F_score,
            val_loss, val_accuracy, val_F_score
        ]

        for key, value in zip(metrics_data.keys(), epoch_data):
            metrics_data[key].append(value)

    print(
        f'train_loss: {metrics_data["train_loss"][-1]:.4f}\ttrain_accuracy: {metrics_data["train_accuracy"][-1]:.4f}\ttrain_F_score: {metrics_data["train_F_score"][-1]:.4f}')
    print(
        f'test_loss: {metrics_data["test_loss"][-1]:.4f}\ttest_accuracy: {metrics_data["test_accuracy"][-1]:.4f}\ttest_F_score: {metrics_data["test_F_score"][-1]:.4f}')
    print(
        f'val_loss: {metrics_data["val_loss"][-1]:.4f}\tval_accuracy: {metrics_data["val_accuracy"][-1]:.4f}, val_F_score: {metrics_data["val_F_score"][-1]:.4f}')

    return metrics_data


if __name__ == '__main__':
    args = get_args()
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    processor = DatasetProcessor('./Dataset', args.max_length, args.batch_size)

    train_loader, test_loader, val_loader, word2vec = processor.generate_all()

    metrics = {}
    if args.network == 'CNN':
        metrics['CNN'] = train(CNN(word2vec),
                               args, train_loader, test_loader, val_loader)
    elif args.network == 'RNN':
        metrics['RNN'] = train(RNN(word2vec),
                               args, train_loader, test_loader, val_loader)
    elif args.network == 'MLP':
        metrics['MLP'] = train(MLP(word2vec),
                               args, train_loader, test_loader, val_loader)
    elif args.network == 'ALL':
        for network in ['CNN', 'RNN', 'MLP']:
            metrics[network] = train(eval(network)(word2vec),
                                     args, train_loader, test_loader, val_loader)

    plot_metrics(metrics)

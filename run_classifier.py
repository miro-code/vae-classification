import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import MNIST

import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import MNIST

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import time
import argparse
from models import MLP
from datasets import LatentDataset

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

def encode_data(model, data_loader):
    # Encode mnist_train and mnist_test into the latent space
    latents = []
    labels = []

    model.eval()
    with torch.no_grad():
        for data, labels in data_loader:
            mean, _ = model.encoder(data)
            latents.append(mean)
            labels.append(labels)

    latents = torch.cat(latents, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    return latents, labels


def train_mlp(train_latents, train_labels, test_latents, test_labels, latent_dim, epochs = 50, lr = 1e-3, batch_size = 256):
    #stratified split of the data
    test_latents, val_latents , test_labels, val_labels = train_test_split(test_latents, test_labels, test_size=0.1, stratify=test_labels)

    print("Number of training samples: ", len(train_latents))
    print("Number of validation samples split from test set for tracking: ", len(val_latents))
    # Load latents into datasets
    train_latent_dataset = LatentDataset(train_latents, train_labels)
    test_latent_dataset = LatentDataset(test_latents, test_labels)
    val_latent_dataset = LatentDataset(val_latents, val_labels)


    

    # Create data loaders for latents
    train_latent_loader = DataLoader(train_latent_dataset, batch_size=batch_size, shuffle=True)
    val_latent_loader = DataLoader(val_latent_dataset, batch_size=batch_size, shuffle=False)
    test_latent_loader = DataLoader(test_latent_dataset, batch_size=batch_size, shuffle=False)


    # Train a simple MLP to classify the latent space
    model = MLP(latent_dim, 10)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for latents, labels in train_latent_loader:
            outputs = model(latents)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}: train loss {loss.item()}")
        
        #get validation loss
        model.eval()
        with torch.no_grad():
            for latents, labels in val_latent_loader:
                outputs = model(latents)
                loss = criterion(outputs, labels)
                print(f"Epoch {epoch + 1}: val loss {loss.item()} (val split from test set)")

            

    # Test the MLP model
    #test_preds = [] TODO
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for latents, labels in test_latent_loader:
            outputs = model(latents)
            _, predicted = torch.max(outputs.data, 1)
            #test_preds.append(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"MLP Test Accuracy: {100 * correct / total}%")

    #create_confusion_matrix(test_preds, test_labels, "rf_confusion_matrix.png") TODO
    
def create_confusion_matrix(predictions, labels, save_path):
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    

def train_rf(train_latents, train_labels, test_latents, test_labels):
    search_params = {
        "n_estimators": [100, 250, 500],
        "min_samples_split": [2, 5, 10],
    }
    clf = GridSearchCV(RandomForestClassifier(), search_params)
    clf.fit(train_latents, train_labels)
    print(f"Best parameters: {clf.best_params_}")
    train_preds = clf.predict(train_latents)
    test_preds = clf.predict(test_latents)
    train_acc = accuracy_score(train_labels, train_preds)
    test_acc = accuracy_score(test_labels, test_preds)
    print(f"Train accuracy: {train_acc}")
    print(f"Test accuracy: {test_acc}")
    create_confusion_matrix(test_preds, test_labels, "rf_confusion_matrix.png")
    

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--epochs", type=int, default=100)
    args.add_argument("--batch_size", type=int, default=256)
    args.add_argument("--latent_dim", type=int, default=20)
    args.add_argument("--lr", type=float, default=1e-3)
    args.add_argument("--dataset", type=str, default="mnist")
    args.add_argument("--data_dir", type=str, default="data/")
    args.add_argument("--n_samples", type=int, default=55)
    args = args.parse_args()

    train_latents = torch.load(f"{args.data_dir}/{args.dataset}/train_latents.pt")
    train_labels = torch.load(f"{args.data_dir}/{args.dataset}/train_labels.pt")
    test_latents = torch.load(f"{args.data_dir}/{args.dataset}/test_latents.pt")
    test_labels = torch.load(f"{args.data_dir}/{args.dataset}/test_labels.pt")

    #do train\test split on the training data
    #random permutation of training array
    indices = torch.randperm(len(train_latents))
    train_latents = train_latents[indices]
    train_labels = train_labels[indices]
    train_labels = train_labels[:args.n_samples]
    train_latents = train_latents[:args.n_samples]

    print("Training random forest")
    start = time.time()
    train_rf(train_latents, train_labels, test_latents, test_labels)
    end = time.time()
    print(f"Time taken: {end - start} seconds")

    print("Training mlp")
    start = time.time()
    print("shapes for debugging", train_latents.shape, train_labels.shape)
    train_mlp(train_latents, train_labels, test_latents, test_labels, args.latent_dim, args.epochs, args.lr, args.batch_size)
    end = time.time()
    print(f"Time taken: {end - start} seconds")

    
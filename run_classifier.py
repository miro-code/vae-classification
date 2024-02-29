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
from models import VAE, MLP
from datasets import LatentDataset


class VAETrainer:
    def __init__(self, model, optimizer, beta):
        self.model = model
        self.optimizer = optimizer
        self.beta = beta

    def loss_function(self, recon, x, mean, logvar):
        recon_loss = nn.MSELoss(reduction='sum')(recon, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss

    def train_step(self, x):
        self.optimizer.zero_grad()
        recon, mean, logvar = self.model(x)
        loss = self.loss_function(recon, x, mean, logvar)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    

def train_vae(batch_size = 256, latent_dim = 20, hidden_channels = [32, 64, 128, 256, 512], lr = 1e-3, beta = 1, vae_epochs = 100, save_freq = 10):
    transform = T.Compose([T.Resize((32, 32)), T.ToTensor()])
    mnist_train = MNIST("data/", train=True, download=True, transform=transform)
    mnist_test = MNIST("data/", train=False, download=True, transform=transform)

    #take 5% of the training data for debugging
    #mnist_train = torch.utils.data.Subset(mnist_train, torch.randperm(len(mnist_train))[:int(len(mnist_train)*0.05)])

    #take 5% of the test data for debugging
    #mnist_test = torch.utils.data.Subset(mnist_test, torch.randperm(len(mnist_test))[:int(len(mnist_test)*0.05)])

    train_loader = DataLoader(mnist_train, batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size, shuffle=True)
    model = VAE(1, latent_dim, (32, 32, 1), hidden_channels)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = VAETrainer(model, optimizer, beta)

    for epoch in range(vae_epochs):
        loss_train = 0
        for X, _ in train_loader:
            loss = trainer.train_step(X)
            loss_train += loss

        print(f"Epoch {epoch + 1}: train loss {loss_train}")
        if(epoch and epoch % save_freq == 0 or epoch == vae_epochs - 1):
            torch.save(model.state_dict(), f"vae_epoch_{epoch}.pt")
    return model


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
    from sklearn.model_selection import train_test_split
    train_latents, val_latents , train_labels, val_labels = train_test_split(train_latents, train_labels, test_size=0.1, stratify=train_labels)

    print("Number of training samples: ", len(train_latents))
    print("Number of validation samples: ", len(val_latents))
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
                print(f"Epoch {epoch + 1}: val loss {loss.item()}")

            

    # Test the MLP model
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for latents, labels in test_latent_loader:
            outputs = model(latents)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"MLP Test Accuracy: {100 * correct / total}%")
    
    #train random forest on the latent space
    

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
    

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--epochs", type=int, default=100)
    args.add_argument("--batch_size", type=int, default=256)
    args.add_argument("--latent_dim", type=int, default=20)
    args.add_argument("--lr", type=float, default=1e-3)
    args.add_argument("--dataset", type=str, default="mnist")
    args.add_argument("--data_dir", type=str, default="data/")
    args = args.parse_args()

    train_latents = torch.load(f"{args.data_dir}/{args.dataset}/train_latents.pt")
    train_labels = torch.load(f"{args.data_dir}/{args.dataset}/train_labels.pt")
    test_latents = torch.load(f"{args.data_dir}/{args.dataset}/test_latents.pt")
    test_labels = torch.load(f"{args.data_dir}/{args.dataset}/test_labels.pt")


    print("Training mlp")
    start = time.time()
    train_mlp(train_latents, train_labels, test_latents, test_labels, args.latent_dim, args.epochs, args.lr, args.batch_size)
    end = time.time()
    print(f"Time taken: {end - start} seconds")

    print("Training random forest")
    start = time.time()
    train_rf(train_latents, train_labels, test_latents, test_labels)
    end = time.time()
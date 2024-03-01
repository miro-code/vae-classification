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

from models import VAE

import argparse
import os


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
    

def train_vae(batch_size, latent_dim, hidden_channels, lr, beta, vae_epochs, save_freq, train_dataset, val_dataset, model_dir, device):
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True)
    model = VAE(1, latent_dim, (32, 32, 1), hidden_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = VAETrainer(model, optimizer, beta)

    for epoch in range(vae_epochs):
        loss_train = 0
        for X, _ in train_loader:
            X = X.to(device)
            loss = trainer.train_step(X)
            loss_train += loss
        print(f"Epoch {epoch + 1}: train loss {loss_train}")

        loss_val = 0
        for X, _ in val_loader:
            X = X.to(device)
            recon, mean, logvar = model(X)
            loss = trainer.loss_function(recon, X, mean, logvar)
            loss_val += loss
        print(f"Epoch {epoch + 1}: val loss {loss_val}")


        if(epoch and epoch % save_freq == 0 or epoch == vae_epochs - 1):
            torch.save(model.state_dict(), f"{model_dir}/vae_epoch_{epoch}.pt")
    return model


def encode_data(model, dataset, batch_size = 256, device = "cpu"):
    # Encode mnist_train and mnist_test into the latent space
    dataloader = DataLoader(dataset, batch_size, shuffle=False)
    latents = []
    labels = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_data, batch_labels in dataloader:
            batch_data = batch_data.to(device)
            mean, _ = model.encoder(batch_data)
            latents.append(mean.detach().cpu())
            labels.append(batch_labels)

    latents = torch.cat(latents, dim=0)
    labels = torch.cat(labels, dim=0)
    return latents, labels


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--dataset", type=str, default="mnist")
    args.add_argument("--batch_size", type=int, default=256)
    args.add_argument("--latent_dim", type=int, default=20)
    args.add_argument("--hidden_channels", type=list, default=[32, 64, 128, 256, 512])
    args.add_argument("--lr", type=float, default=1e-3)
    args.add_argument("--beta", type=float, default=1)
    args.add_argument("--epochs", type=int, default=100)
    args.add_argument("--save_freq", type=int, default=10)
    args.add_argument("--data_dir", type=str, default="data/")
    args.add_argument("--model_dir", type=str, default="models/")
    args = args.parse_args()
    print("starting vae training with the following arguments:")
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if(args.dataset == "mnist"):
        transform = T.Compose([T.Resize((32, 32)), T.ToTensor()])
        train_dataset = MNIST(args.data_dir, train=True, download=True, transform=transform)
        test_dataset = MNIST(args.data_dir, train=False, download=True, transform=transform)

    #make sure dir exists
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(f"{args.data_dir}/{args.dataset}", exist_ok=True)
    
    vae = train_vae(args.batch_size, args.latent_dim, args.hidden_channels, args.lr, args.beta, args.epochs, args.save_freq, train_dataset, test_dataset, args.model_dir, device)
    train_latens, train_labels = encode_data(vae, train_dataset, device=device)
    test_latens, test_labels = encode_data(vae, test_dataset, device=device)

    #save to data_dir
    torch.save(train_latens, f"{args.data_dir}/{args.dataset}/train_latents.pt")
    torch.save(train_labels, f"{args.data_dir}/{args.dataset}/train_labels.pt")
    torch.save(test_latens, f"{args.data_dir}/{args.dataset}/test_latents.pt")
    torch.save(test_labels, f"{args.data_dir}/{args.dataset}/test_labels.pt")

if __name__ == "__main__":
    main()
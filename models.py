
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import MNIST

class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim, hidden_channels):
        #Todo make work for not 32x32 images: change self.mu and self.logvar input dims
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        
        layers = []
        for channel in hidden_channels:
            layers.append(nn.Conv2d(in_channels, channel, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(channel))
            layers.append(nn.ReLU())
            in_channels = channel

        self.encoder = nn.Sequential(*layers)
        self.mu = nn.Linear(hidden_channels[-1], latent_dim)
        self.logvar = nn.Linear(hidden_channels[-1], latent_dim)

    def forward(self, x, training=True):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_channels):
        super(Decoder, self).__init__()

        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        H, W, C = self.output_dim
        factor = 2 ** len(self.hidden_channels)
        assert (
            H % factor == W % factor == 0
        ), f"output_dim must be a multiple of {factor}"
        H, W = H // factor, W // factor

        self.fc = nn.Linear(latent_dim, H * W * self.hidden_channels[-1])

        self.reshape = lambda x: x.view(-1, self.hidden_channels[-1], H, W)

        layers = []
        in_channels = hidden_channels[-1]
        for channel in reversed(hidden_channels[:-1]):
            #CHANGE Double check the padding and output padding
            layers.append(nn.ConvTranspose2d(in_channels, channel, kernel_size=3, stride=2, padding=1, output_padding=1))
            layers.append(nn.BatchNorm2d(channel))
            layers.append(nn.ReLU())
            in_channels = channel
        layers.append(nn.ConvTranspose2d(in_channels, output_dim[2], kernel_size=3, stride=2, padding=1, output_padding=1))
        layers.append(nn.Sigmoid())
        
        self.convolutions = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc(x)
        x = nn.functional.relu(x)
        x = self.reshape(x)
        x = self.convolutions(x)
        return x

class VAE(nn.Module):
    #reimplements the VAE from https://github.com/probml/probml-utils/blob/main/probml_utils/conv_vae_flax_utils.py in pytorch
    def __init__(self, in_channels, latent_dim, output_dim, hidden_channels, variational=True):
        super(VAE, self).__init__()
        self.variational = variational
        self.encoder = Encoder(in_channels, latent_dim, hidden_channels)
        self.decoder = Decoder(latent_dim, output_dim, hidden_channels)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, training=True):
        mean, logvar = self.encoder(x)
        if self.variational:
            z = self.reparameterize(mean, logvar)
        else:
            z = mean
        recon = self.decoder(z)
        return recon, mean, logvar

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

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
from torch.utils.data import Dataset

class LatentDataset(Dataset):
    def __init__(self, latents, labels):
        self.latents = latents
        self.labels = labels

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        latent = self.latents[idx]
        label = self.labels[idx]
        return latent, label

def mnist_demo():
    batch_size = 256
    latent_dim = 20
    hidden_channels = [32, 64, 128, 256, 512]
    lr = 1e-3
    beta = 1
    vae_epochs = 10
    

    transform = T.Compose([T.Resize((32, 32)), T.ToTensor()])
    mnist_train = MNIST("data/", train=True, download=True, transform=transform)
    mnist_test = MNIST("data/", train=False, download=True, transform=transform)

    #take 5% of the training data
    #mnist_train = torch.utils.data.Subset(mnist_train, torch.randperm(len(mnist_train))[:int(len(mnist_train)*0.05)])

    #take 5% of the test data
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

    

    # Encode mnist_train and mnist_test into the latent space
    train_latents = []
    train_labels = []
    test_latents = []
    test_labels = []

    model.eval()
    with torch.no_grad():
        for data, labels in train_loader:
            mean, _ = model.encoder(data)
            train_latents.append(mean)
            train_labels.append(labels)

        for data, labels in test_loader:
            mean, _ = model.encoder(data)
            test_latents.append(mean)
            test_labels.append(labels)

    train_latents = torch.cat(train_latents, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    test_latents = torch.cat(test_latents, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    # Load latents into datasets
    train_latent_dataset = LatentDataset(train_latents, train_labels)
    test_latent_dataset = LatentDataset(test_latents, test_labels)



    mlp_epochs = 20
    mlp_lr = 1e-3
    mlp_batch_size = 256

    # Create data loaders for latents
    train_latent_loader = DataLoader(train_latent_dataset, batch_size=mlp_batch_size, shuffle=True)
    test_latent_loader = DataLoader(test_latent_dataset, batch_size=mlp_batch_size, shuffle=False)


    # Train a simple MLP to classify the latent space
    mlp_model = MLP(latent_dim, 10)
    mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=mlp_lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(mlp_epochs):
        mlp_model.train()
        for latents, labels in train_latent_loader:
            outputs = mlp_model(latents)
            loss = criterion(outputs, labels)
            mlp_optimizer.zero_grad()
            loss.backward()
            mlp_optimizer.step()

    # Test the MLP model
    with torch.no_grad():
        mlp_model.eval()
        correct = 0
        total = 0
        for latents, labels in test_latent_loader:
            outputs = mlp_model(latents)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"MLP Test Accuracy: {100 * correct / total}%")
    
    #train random forest on the latent space
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    clf = RandomForestClassifier()
    clf.fit(train_latents, train_labels)
    preds = clf.predict(test_latents)
    accuracy = accuracy_score(test_labels, preds)
    print(f"Random Forest Test Accuracy: {accuracy * 100}%")

if __name__ == "__main__":
    mnist_demo()
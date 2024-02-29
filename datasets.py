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

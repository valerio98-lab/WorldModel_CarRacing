import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch import optim
from torchvision import transforms, datasets
import tqdm.auto as tqdm
import os

from vae import VAE

torch.manual_seed(42)



class trainVAE(nn.Module):
    def __init__(self, input_channels, latent_dim=32, batch_size=32, epochs=10):
        super(trainVAE, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_channels = 3
        self.epochs = epochs
        self.latent_dim=latent_dim
        self.batch_size = batch_size
        self.vae = VAE(input_channels, latent_dim).to(self.device)

        self.optimizer = optim.Adam(self.vae.parameters())
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64))
        ])

        self.dataset_train = datasets.MNIST(root='./data', train=True, transform=self.transform, download=True)
        self.dataset_test = datasets.MNIST(root='./data', train=False, transform=self.transform, download=True)
        self.dataset_val = ...

        self.train_loader = data.DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True)
        self.test_loader = data.DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False)
        self.val_loader = data.DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False)


    def train(self, epoch):
        self.vae.train()
        train_loss = 0
        print("check batch_size", self.train_loader.batch_size)
        for _, (batch, _) in tqdm.tqdm(enumerate(self.train_loader)):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            recon, mu, sigma = self.vae(batch)
            loss = self.vae.loss_function(recon, batch, mu, sigma)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
        
        print(f"Epoch: {epoch}, Loss: {train_loss / len(self.train_loader.dataset)}")
        
        torch.cuda.empty_cache()


    def test(self, epoch):
        self.vae.eval()
        test_loss = 0
        with torch.no_grad():
            for _, (batch, _) in tqdm.tqdm(enumerate(self.test_loader)):
                batch = batch.to(self.device)
                recon, mu, sigma = self.vae(batch)
                loss = self.vae.loss_function(recon, batch, mu, sigma)
                test_loss += loss
        

        print(f"Epoch: {epoch}, Loss: {test_loss / len(self.test_loader.dataset)}")
        torch.cuda.empty_cache()
        
        return test_loss / len(self.test_loader.dataset)



    def validation(self, epoch):
        self.vae.eval()
        val_loss = 0
        with torch.no_grad():
            for _, (batch, _) in tqdm.tqdm(enumerate(self.val_loader)):
                batch = batch.to(self.device)
                recon, mu, sigma = self.vae(batch)
                val_loss += self.vae.loss_function(recon, batch, mu, sigma)

        print(f"Epoch: {epoch}, Loss: {val_loss / len(self.val_loader.dataset)}")
        torch.cuda.empty_cache()
        
        return val_loss / len(self.val_loader.dataset)


    def train_model(self):
        for epoch in range(self.epochs):
            self.train(epoch)
            self.test(epoch)
            #val_loss = self.validation(epoch)
            #self.scheduler.step(val_loss)
            self.save_checkpoint(epoch)
            torch.cuda.empty_cache()



    def save_checkpoint(self, epoch):
        ## Save the model, optimizer, scheduler and epoch
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.vae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, './checkpoints/checkpoint.tar')
    

    def load(self):
        checkpoint = torch.load('./checkpoints/checkpoint.tar', map_location=self.device, weights_only=True)
        self.vae.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epochs = checkpoint['epoch']


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt

#     def show_reconstructions(model, dataloader, device):
#         model.eval()
#         with torch.no_grad():
#             for (batch, _) in dataloader:
#                 batch = batch.to(device)
#                 recon, _, _ = model(batch)
#                 # Mostra le immagini originali e quelle ricostruite
#                 fig, axes = plt.subplots(2, 10, figsize=(15, 3))
#                 for i in range(10):
#                     # Mostra le immagini originali
#                     axes[0, i].imshow(batch[i].cpu().permute(1, 2, 0).squeeze(), cmap='gray')
#                     axes[0, i].axis('off')
#                     # Mostra le ricostruzioni
#                     axes[1, i].imshow(recon[i].cpu().permute(1, 2, 0).squeeze(), cmap='gray')
#                     axes[1, i].axis('off')
#                 plt.show()
#                 break

#     # Mostra le ricostruzioni usando il dataloader di test
#     model = trainVAE(input_channels=1, latent_dim=32, batch_size=32, epochs=30)
#     model.train_model()
#     model.load()
#     show_reconstructions(model.vae, model.test_loader, model.device)
    
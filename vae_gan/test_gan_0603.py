# Define the import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from tqdm.auto import tqdm
import time
import math
from scipy.ndimage import rotate as nd_rotate

# Create directory for saving all the calculated results
os.makedirs('results', exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#--------------------------#
# Data Loading and Setup   #
#--------------------------#

def load_mnist_data(batch_size=64, download=True):
    """
    Load MNIST dataset with proper transforms.
    Args:
        batch_size (int): The batch size for the data loaders.
        download (bool): Whether to download the dataset.
    Returns:
        trainset (torchvision.datasets.MNIST): The training dataset.
    """

    in_labels = [1, 8]
    out_labels = list(set(range(10)) - set(in_labels))
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load training dataset
    trainset = torchvision.datasets.MNIST(
        root='./MNIST',
        train=True,
        download=download,
        transform=transform
    )

    # Load test dataset
    testset = torchvision.datasets.MNIST(
        root='./MNIST',
        train=False,
        download=download,
        transform=transform
    )
    # Create datasets for out_labels
    out_trainset = torch.utils.data.Subset(trainset, [i for i, (x, y) in enumerate(trainset) if y in out_labels])
    out_testset = torch.utils.data.Subset(testset, [i for i, (x, y) in enumerate(testset) if y in out_labels])
    # Create data loaders
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    out_trainloader = torch.utils.data.DataLoader(out_trainset, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True)
    out_testloader = torch.utils.data.DataLoader(out_testset, batch_size=batch_size, shuffle=False, num_workers=4,pin_memory=True)

    return trainset, testset, trainloader, testloader, out_trainset, out_testset, out_trainloader, out_testloader

def load_notmnist(batch_size=64):
    """
    Load the notMNIST dataset as an out-of-distribution test set.
    This assumes you have notMNIST data in the correct format.
    If not, it will provide a warning.
    Args:
        batch_size (int): The batch size for the data loaders.
    Returns:
        notmnist_data (torchvision.datasets.ImageFolder): The notMNIST dataset.
        notmnist_loader (torch.utils.data.DataLoader): The data loader for the notMNIST dataset.
    """
    try:
        # Try to load notMNIST if available
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        notmnist_data =torchvision.datasets.FashionMNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )

        notmnist_loader = torch.utils.data.DataLoader(
            notmnist_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        # Extract the test set and normalize it for OoD testing
        OoD = torch.stack([notmnist_data[i][0] for i in range(len(notmnist_data))])
        OoD = OoD / 255.0
        OoD_dataset = torch.utils.data.TensorDataset(OoD)
        OoD_loader = torch.utils.data.DataLoader(
            OoD_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        return notmnist_data, notmnist_loader,OoD_dataset,OoD_loader

    except:
        # If notMNIST is not available, print a warning
        # TODO: Implement OOD testing with notMNIST dataset
        print("""
        Warning: notMNIST dataset not found.
        For out-of-distribution testing, please download the notMNIST dataset from:
        https://github.com/davidflanagan/notMNIST-to-MNIST

        For now, this is not implemented yet.
        """)
        return None, None,None,None

#----------------------------#
# Core Model Components     #
#----------------------------#

def exp_evidence(logits):
    """Convert logits to evidence through exponential function with clamping.
    Args:
        logits (torch.Tensor): The logits of the model.
    Returns:
        evidence (torch.Tensor): The evidence of the model.
    """
    return torch.exp(torch.clamp(logits, -50, 50))

def KL(alpha):
    """Calculate KL divergence between alpha (Dirichlet) and uniform Dirichlet.
    Args:
        alpha (torch.Tensor): The parameters of the Dirichlet distribution.
    Returns:
        kl (torch.Tensor): The KL divergence between the Dirichlet and uniform Dirichlet.
    """
    K = alpha.size(-1)
    beta = torch.ones((1, K), dtype=torch.float32).to(alpha.device)

    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)

    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.special.digamma(S_alpha)
    dg1 = torch.special.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def compute_misleading_uncertainty(alpha, y):
    """
    Compute uncertainty for misclassified samples.
    Args:
        alpha: Dirichlet parameters (batch_size, num_classes)
        y: One-hot encoded labels (batch_size, num_classes)
    Returns:
        Dirichlet parameters for non-target classes
    """
    K = y.size(-1)
    # Identify non-true classes
    # A crucial step to compute the misleading uncertainty
    alpha_mask = (y == 0)

    # Collect alpha values for non-true classes for each sample
    misleading_alphas = []
    for i in range(alpha.size(0)):
        # Get alpha values only for non-target classes
        sample_mask = alpha_mask[i]
        if torch.sum(sample_mask) > 0:  # At least one non-target class
            misleading_alpha = alpha[i, sample_mask]
            misleading_alphas.append(misleading_alpha)

    if misleading_alphas:
        # Stack all misleading alphas
        return torch.stack(misleading_alphas)
    else:
        # Return empty tensor
        return torch.tensor([], device=alpha.device)

def uncertainty_loss(y, evidence, real_p, fake_p):
    """
    Compute the full uncertainty-aware loss function.
    Args:
        y: One-hot encoded labels (batch_size, num_classes)
        evidence: Evidence for each class from model (batch_size, num_classes)
        real_p: Probabilities for real data (batch_size, num_classes)
        fake_p: Probabilities for fake data (batch_size, num_classes)
    Returns:
        Total loss, classification loss component, KL loss component, uncertainty
    """
    K = y.size(1)  # Number of classes

    # Classification loss
    real_loss = torch.sum(-y * torch.log(real_p + 1e-5), dim=1)
    fake_loss = torch.sum(-y * torch.log(1.0 - fake_p + 1e-5), dim=1)
    classification_loss = torch.mean(real_loss + fake_loss)

    # Computing alpha
    alpha = evidence + 1

    # Compute uncertainty
    S = torch.sum(alpha, dim=1, keepdim=True)
    uncertainty = K / S

    # KL divergence for non-target classes
    misleading_alpha = compute_misleading_uncertainty(alpha, y)

    kl_loss = 0
    if misleading_alpha.numel() > 0:
        # Weight KL loss by expected probability of misclassification
        # (1 - p_k) where p_k is the probability of the true class
        true_class_prob = torch.sum(y * (alpha / S), dim=1)
        kl_weight = torch.clamp(1.0 - true_class_prob, min=0.01, max=0.99)

        kl_div = KL(misleading_alpha)
        kl_loss = torch.mean(kl_weight * kl_div)

    # Total loss
    total_loss = classification_loss + 0.1 * kl_loss  

    return total_loss, classification_loss, kl_loss, uncertainty

#----------------------------#
# Model Architectures       #
#----------------------------#

class Encoder(nn.Module):
    def __init__(self, latent_dim=100):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)

        self.fc_loc = nn.Linear(50 * 4 * 4, latent_dim)
        self.fc_scale = nn.Linear(50 * 4 * 4, latent_dim)

    def forward(self, x):
        # Shape (batch_size, 1, 28, 28)
        x = x.view(x.size(0), 1, 28, 28)

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = torch.flatten(x, start_dim=1)

        # Generate mean and variance for latent distribution
        loc = self.fc_loc(x)
        scale = F.softplus(self.fc_scale(x)) + 1e-5

        m = dist.MultivariateNormal(loc, torch.diag_embed(scale))

        # Sample from the distribution with reparameterization trick
        code = m.rsample()

        return code, loc, scale

class Decoder(nn.Module):
    def __init__(self, latent_dim=100):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(latent_dim, 7*7*256)
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 1, kernel_size=5, stride=2, padding=2,output_padding=1)
        #self.batch_norm1 = nn.BatchNorm2d(128)
        #self.batch_norm2 = nn.BatchNorm2d(1)

    def forward(self, z):
        x = self.fc1(z)
        x = F.relu(x)
        x = x.view(-1, 256, 7, 7)

        x = F.relu(self.deconv1(x))
        x = torch.tanh(self.deconv2(x))

        return x

class Generator(nn.Module):
    def __init__(self, latent_dim=100, noise_dim=2):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.noise_dim = noise_dim

        self.fc1 = nn.Linear(latent_dim + noise_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, latent_dim)

    def forward(self, code):
        batch_size = code.size(0)

        # Generate random noise
        noise = torch.randn(batch_size, self.noise_dim, device=code.device)

        # Concatenate noise with latent code
        x = torch.cat((noise, code), dim=1)

        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.2)

        # Generate standard deviation
        std = F.softplus(self.fc4(x)) + 1e-5

        return std

class LatentDiscriminator(nn.Module):
    def __init__(self, latent_dim=100):
        super(LatentDiscriminator, self).__init__()

        self.fc1 = nn.Linear(latent_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.2)
        x = self.fc4(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, K=10):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)

        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, K)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Shape (batch_size, 1, 28, 28)
        x = x.view(x.size(0), 1, 28, 28)

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class Autoencoder(nn.Module):
    def __init__(self, latent_dim=100):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.generator = Generator(latent_dim)
        self.discriminator = Discriminator()  
        self.latent_disc = LatentDiscriminator(latent_dim)

        # Define prior distribution
        self.prior = None
        self.latent_dim = latent_dim

    def make_prior(self, device=None):
        loc = torch.zeros(self.latent_dim, device=device)
        scale = torch.ones(self.latent_dim, device=device)
        return dist.MultivariateNormal(loc, torch.diag(scale))

    def wloss(self,logits, maximize=True):
        labels = torch.ones_like(logits) if maximize else torch.zeros_like(logits)
        return torch.mean(F.binary_cross_entropy_with_logits(logits, labels))

    def forward(self, X):
        if self.prior is None or self.prior.loc.device != X.device:
            self.prior = self.make_prior(device=X.device)

        # Encode input
        code, _, _ = self.encoder(X)

        # Generate std and sample from latent space
        std = torch.diag_embed(self.generator(code)) + 1e-3
        pdf = dist.MultivariateNormal(code, std)
        fake = pdf.sample()

        # Run Discriminator in latent space
        rlogits = self.latent_disc(code)
        r_p = torch.sigmoid(rlogits)

        flogits = self.latent_disc(fake)
        f_p = torch.sigmoid(flogits)

        # Reconstruct images
        recon = self.decoder(code)
        X_fake = self.decoder(fake)

        # Run discriminator on reconstructed and fake images
        real_logits = self.discriminator(X)
        real_p = torch.sigmoid(real_logits)

        fake_logits = self.discriminator(X_fake)
        fake_p = torch.sigmoid(fake_logits)

        # KL divergence with prior
        kl = -torch.mean(self.prior.log_prob(code))
        kl_fake = -torch.mean(self.prior.log_prob(fake))

        # discriminator loss in the image space
        loss_disc = torch.mean(-torch.log(real_p + 1e-8)) + torch.mean(-torch.log(1 - fake_p + 1e-8))

        # Latent discriminator loss
        latent_disc_loss = torch.mean(-torch.log(r_p + 1e-5)) + torch.mean(-torch.log(1 - f_p + 1e-5))

        # generator loss
        loss_gen = torch.mean(-torch.log(1 - fake_p + 1e-8)) + torch.mean(-torch.log(f_p + 1e-8))
        
        # Reconstruction loss
        rec_loss = torch.mean(torch.sum((recon - X) ** 2, dim=1) + 1e-4) + 0.1 * kl
        
        # Full reconstruction loss
        rec_loss += self.wloss(flogits, False)

        return X, X_fake, code, recon, rec_loss, latent_disc_loss, loss_disc, loss_gen

class UncertaintyLeNet(nn.Module):
    def __init__(self, K=10, latent_dim=50):
        super(UncertaintyLeNet, self).__init__()
        self.K = K
        self.latent_dim = latent_dim

        self.autoencoder = Autoencoder(latent_dim)
        self.discriminator = Discriminator(K=K)

    def forward(self, X, Y=None):
        logits = self.discriminator(X)
        evidence = exp_evidence(logits)
        alpha = evidence + 1

        # Calculate uncertainty
        uncertainty = self.K / torch.sum(alpha, dim=1, keepdim=True)

        # Calculate probabilities
        S = torch.sum(alpha, dim=1, keepdim=True)
        probs = alpha / S

        # Generate fake samples if the autoencoder is not None
        if hasattr(self, 'autoencoder') and self.autoencoder is not None:
            _, X_fake, _, _, _, _,_,_ = self.autoencoder(X)
            fake_logits = self.discriminator(X_fake)
            fake_p = torch.sigmoid(fake_logits)
        else:
            X_fake = None
            fake_p = torch.zeros_like(probs).to(X.device)

        # Calculate real probabilities
        real_p = torch.sigmoid(logits)

        pred = torch.argmax(logits, dim=1)

        if Y is not None:
            truth = torch.argmax(Y, dim=1)
            match = (pred == truth).float().view(-1, 1)
            acc = torch.mean(match)

            # Get evidence statistics
            total_evidence = torch.sum(evidence, dim=1, keepdim=True)
            mean_ev = torch.mean(total_evidence)

            # Get evidence statistics
            mean_ev_succ = torch.tensor(0.0, device=X.device)
            if torch.sum(match) > 0:
                mean_ev_succ = torch.mean(total_evidence[match == 1])

            mean_ev_fail = torch.tensor(0.0, device=X.device)
            if torch.sum(1 - match) > 0:
                mean_ev_fail = torch.mean(total_evidence[match == 0])
        else:
            # Placeholders!
            # TODO: This part may need to be implemented!
            truth = pred
            match = torch.ones_like(pred, dtype=torch.float32).view(-1, 1)
            acc = torch.tensor(0.0, device=X.device)
            mean_ev = torch.tensor(0.0, device=X.device)
            mean_ev_succ = torch.tensor(0.0, device=X.device)
            mean_ev_fail = torch.tensor(0.0, device=X.device)

        return (real_p, fake_p, probs, pred, truth, match, acc, evidence,
                mean_ev, mean_ev_succ, mean_ev_fail, X_fake, uncertainty)

#----------------------------#
# Training Procedures        #
#----------------------------#

def train_vae_gan(autoencoder, train_loader, num_epochs=10, lr=0.001):
    """
    Train the VAE-GAN component for generating OOD samples.
    Args:
        autoencoder: Autoencoder model
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        lr: Learning rate
    """
    device = next(autoencoder.parameters()).device

    ae_optimizer = optim.Adam(list(autoencoder.encoder.parameters()) +
                             list(autoencoder.decoder.parameters()), lr=lr)
    disc_optmizer = optim.RMSprop(autoencoder.discriminator.parameters(), lr=lr)
    gen_optimizer = optim.RMSprop(autoencoder.generator.parameters(), lr=lr)
    latent_disc_optimizer = optim.RMSprop(autoencoder.latent_disc.parameters(), lr=lr)

    history = {
        'rec_loss': [],
        'disc_loss': [],
        'latent_disc_loss': []
    }

    print("Training VAE-GAN component...")
    for epoch in range(num_epochs):
        # Initialize epoch losses
        epoch_rec_loss = 0
        epoch_disc_loss = 0
        epoch_latent_disc_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for data, _ in pbar:
            batch_size = data.size(0)
            data = data.to(device)

            ae_optimizer.zero_grad()
            disc_optmizer.zero_grad()
            gen_optimizer.zero_grad()
            latent_disc_optimizer.zero_grad()

            _, Xfake, code, recon, rec_loss, latent_disc_loss, loss_disc, loss_gen = autoencoder(data)

            latent_disc_loss.backward(retain_graph=True)
            latent_disc_optimizer.step()

            gen_optimizer.zero_grad()
            _, Xfake, code, recon, _, _, _, loss_gen = autoencoder(data)
            loss_gen.backward()
            gen_optimizer.step()

            ae_optimizer.zero_grad()
            _, Xfake, code, recon, rec_loss, _, _, _ = autoencoder(data)
            rec_loss.backward()
            ae_optimizer.step()

            disc_optmizer.zero_grad()
            _, Xfake, code, recon, _, _, loss_disc, _ = autoencoder(data)
            loss_disc.backward()
            disc_optmizer.step()

            epoch_rec_loss += rec_loss.item()
            epoch_disc_loss += loss_disc.item()
            epoch_latent_disc_loss += latent_disc_loss.item()

            pbar.set_postfix({
                'rec_loss': rec_loss.item(),
                'disc_loss': loss_disc.item(),
                'latent_disc_loss': latent_disc_loss.item()
            })

        avg_rec_loss = epoch_rec_loss / len(train_loader)
        avg_disc_loss = epoch_disc_loss / len(train_loader)
        avg_latent_disc_loss = epoch_latent_disc_loss / len(train_loader)

        history['rec_loss'].append(avg_rec_loss)
        history['disc_loss'].append(avg_disc_loss)
        history['latent_disc_loss'].append(avg_latent_disc_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Rec Loss: {avg_rec_loss:.4f}, "
              f"Disc Loss: {avg_disc_loss:.4f}, "
              f"Latent Disc Loss: {avg_latent_disc_loss:.4f}")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            visualize_vae_samples(autoencoder, data[:10], epoch+1)

    return history

def visualize_vae_samples(autoencoder, data, epoch):
    """
    Visualize original, reconstructed, and generated samples.
    Args:
        autoencoder: Autoencoder model
        data: Data to visualize
        epoch: Current epoch
    """
    device = next(autoencoder.parameters()).device
    autoencoder.eval()

    with torch.no_grad():
        _, Xfake, _, recon, _, _, _, _ = autoencoder(data)

    # Visualize original, reconstructed, and generated samples
    num_samples = data.size(0)
    num_cols = min(10, num_samples)
    num_rows = 3
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6))

    def denormalize(x):
        return (x * 0.5 + 0.5).cpu().numpy()

    # Plot original images
    for i in range(min(10, data.size(0))):
        if data.size(0) >= 10:
            ax = axes[0, i]
        else:
            ax = axes[0, i]
        img = denormalize(data[i]).squeeze()
        ax.imshow(img)
        ax.axis('off')
        if i == 0:
            ax.set_title('Original')

    # Plot reconstructed images
    for i in range(min(10, recon.size(0))):
        if recon.size(0) >= 10:
            ax = axes[1, i]
        else:
            ax = axes[1, i]
        img = denormalize(recon[i]).squeeze()
        ax.imshow(img)
        ax.axis('off')
        if i == 0:
            ax.set_title('Reconstructed')

    # Plot generated fake images
    for i in range(min(10, Xfake.size(0))):
        if Xfake.size(0) >= 10:
            ax = axes[2, i]
        else:
            ax = axes[2, i]
        img = denormalize(Xfake[i]).squeeze()
        ax.imshow(img)
        ax.axis('off')
        if i == 0:
            ax.set_title('Generated')

    plt.tight_layout()
    plt.savefig(f'results/vae_samples_epoch_{epoch}.png')
    plt.close(fig)

def train_model(model, train_loader, test_loader, num_epochs=50, lr=0.001):
    """
    Train the full uncertainty-aware model.
    Args:
        model: UncertaintyLeNet model
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        num_epochs: Number of training epochs
        lr: Learning rate
    """
    device = next(model.parameters()).device

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

    history = {
        'train_acc': [],
        'train_ev_succ': [],
        'train_ev_fail': [],
        'test_acc': [],
        'test_ev_succ': [],
        'test_ev_fail': [],
        'total_loss': [],
        'class_loss': [],
        'kl_loss': []
    }

    print("Training uncertainty-aware classifier...")
    for epoch in range(num_epochs):
        model.train()
        running_total_loss = 0.0
        running_class_loss = 0.0
        running_kl_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for data, targets in pbar:
            data, targets = data.to(device), targets.to(device)
            Y = F.one_hot(targets, num_classes=10).float()

            optimizer.zero_grad()

            real_p, fake_p, prob, pred, truth, match, acc, evidence, mean_ev, mean_ev_succ, mean_ev_fail, X_fake, uncertainty = model(data, Y)

            total_loss, class_loss, kl_loss, _ = uncertainty_loss(Y, evidence, real_p, fake_p)

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_total_loss += total_loss.item()
            running_class_loss += class_loss.item()
            running_kl_loss += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else 0

            pbar.set_postfix({
                'loss': total_loss.item(),
                'acc': acc.item()
            })

        # Evaluation
        model.eval()
        with torch.inference_mode(): # a littttttle bit faster than torch.no_grad()
            sample_size = min(10000, len(train_loader.dataset))
            indices = torch.randperm(len(train_loader.dataset))[:sample_size]

            train_sample = torch.stack([train_loader.dataset[i][0] for i in indices]).to(device)
            train_targets = torch.tensor([train_loader.dataset[i][1] for i in indices]).to(device)
            train_labels = F.one_hot(train_targets, num_classes=10).float()

            _, _, _, _, _, _, train_acc, _, _, train_succ, train_fail, _, _ = model(train_sample, train_labels)

            test_data = torch.stack([test_loader.dataset[i][0] for i in range(len(test_loader.dataset))]).to(device)
            test_targets = torch.tensor([test_loader.dataset[i][1] for i in range(len(test_loader.dataset))]).to(device)
            test_labels = F.one_hot(test_targets, num_classes=10).float()

            _, _, _, _, _, _, test_acc, _, _, test_succ, test_fail, _, _ = model(test_data, test_labels)

        avg_total_loss = running_total_loss / len(train_loader)
        avg_class_loss = running_class_loss / len(train_loader)
        avg_kl_loss = running_kl_loss / len(train_loader)

        history['train_acc'].append(train_acc.item())
        history['train_ev_succ'].append(train_succ.item())
        history['train_ev_fail'].append(train_fail.item())
        history['test_acc'].append(test_acc.item())
        history['test_ev_succ'].append(test_succ.item())
        history['test_ev_fail'].append(test_fail.item())
        history['total_loss'].append(avg_total_loss)
        history['class_loss'].append(avg_class_loss)
        history['kl_loss'].append(avg_kl_loss)

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"Loss: {avg_total_loss:.4f} (Class: {avg_class_loss:.4f}, KL: {avg_kl_loss:.4f})")
        print(f"Training Acc: {train_acc.item():.4f} (Evidence - Success: {train_succ.item():.4f}, Fail: {train_fail.item():.4f})")
        print(f"Testing Acc: {test_acc.item():.4f} (Evidence - Success: {test_succ.item():.4f}, Fail: {test_fail.item():.4f})")

        # Save model checkpoint each ten epochs
        # TODO: Set up your preferred checkpoint saving strategy!
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, f'results/model_checkpoint_epoch_{epoch+1}.pt')

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history
    }, 'results/model_final.pt')

    return model, history

#----------------------------#
# Evaluation Functions       #
#----------------------------#

def plot_training_history(history):
    """
    Plot training metrics from history.
    Args:
        history: History dictionary from training
    """
    plt.figure(figsize=(15, 10))

    # Plot accuracy
    plt.subplot(2, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['test_acc'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()

    # Plot evidence for successful and failed predictions
    plt.subplot(2, 2, 2)
    plt.plot(history['train_ev_succ'], label='Evidence (Success)')
    plt.plot(history['train_ev_fail'], label='Evidence (Fail)')
    plt.xlabel('Epoch')
    plt.ylabel('Evidence')
    plt.title('Evidence Values')
    plt.legend()

    # Plot losses
    plt.subplot(2, 2, 3)
    plt.plot(history['total_loss'], label='Total Loss')
    plt.plot(history['class_loss'], label='Classification Loss')
    plt.plot(history['kl_loss'], label='KL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()

    # Save the figure
    plt.tight_layout()
    plt.savefig('results/training_history.png')
    plt.close()

def get_image(data):
    """
    Convert batch of images into a grid.
    Args:
        data: numpy array of shape (batch_size, channels, height, width) or (batch_size, height, width)
    Returns:
        grid_image: numpy array of shape (n*h, m*w) where n,m are calculated from batch_size
    """
    # Remove channel dimension if present
    if len(data.shape) == 4:
        data = data.squeeze(1)
    
    # Calculate grid dimensions
    n = int(np.sqrt(data.shape[0]))
    m = int(np.ceil(data.shape[0]/n))
    
    # Create output array
    I = np.zeros((n*28, m*28))
    
    # Fill the grid with images
    for i in range(n):
        for j in range(m):
            if i*m + j < data.shape[0]:
                I[i*28:(i+1)*28, j*28:(j+1)*28] = data[i*m + j]
    
    return I

def plot_original_and_generated_images(model, testset, start_idx=100, end_idx=200):
    """
    Plot original and generated images using the full uncertainty-aware model.
    Args:
        model: UncertaintyLeNet model
        testset: Test dataset
        start_idx: Starting index for images
        end_idx: Ending index for images
    """
    # Get a batch of images
    images = [testset[i][0] for i in range(start_idx, end_idx)]
    images = torch.stack(images)  # Stack images into a single tensor
    
    device = next(model.parameters()).device
    model.eval()  # Set model to evaluation mode
    
    # Move images to device and prepare dummy labels
    images = images.to(device)
    dummy_labels = torch.zeros(images.shape[0], 10, dtype=torch.float32).to(device)
    
    # Generate fake images using the full model
    with torch.no_grad():
        _, _, _, _, _, _, _, _, _, _, _, X_fake, _ = model(images, dummy_labels)
    
    # Process original images
    images = images.cpu().numpy()
    org = get_image(images)
    
    # Process generated/fake images
    X_fake = X_fake.cpu().numpy()
    fake = get_image(X_fake)
    
    # Create figure
    plt.figure(figsize=(10, 5))
    
    # Plot original images
    plt.subplot(1, 2, 1)
    plt.imshow(org)  # Remove vmin/vmax to use data range
    plt.title("Original Images")
    plt.axis('off')
    
    # Plot generated images
    plt.subplot(1, 2, 2)
    plt.imshow(fake)  # Remove vmin/vmax to use data range
    plt.title("Generated Images")
    plt.axis('off')
    
    # Adjust layout and save with high DPI
    plt.tight_layout()
    plt.savefig('results/Original_and_Generated_images.png', 
                bbox_inches='tight', 
                dpi=300,
                facecolor='white',
                edgecolor='none')
    plt.close()

def rotate_img(x, deg):
    return torch.from_numpy(nd_rotate(x.reshape(28, 28), deg, reshape=False).ravel()).float()

def rotating_image_classification(img, model, keep_prob=None, prob_threshold=0.5):
    Mdeg = 180
    Ndeg = int(Mdeg / 10) + 1
    ldeg = []
    lp = []
    lu = []
    scores = np.zeros((1, 10))  # Assuming 10 classes
    rimgs = np.zeros((28, 28 * Ndeg))
    
    device = next(model.parameters()).device
    model.eval()
    
    # Ensure input image is properly normalized
    if img.max() > 1.0:
        img = (img - 0.5) / 0.5  # Denormalize first
    img = img.numpy()
    
    for i, deg in enumerate(np.linspace(0, Mdeg, Ndeg)):
        # Rotate and normalize the image
        nimg = rotate_img(img, deg)
        nimg = nimg.numpy()
        nimg = np.clip(nimg, -1, 1)  # Clip to [-1, 1] range
        rimgs[:, i * 28:(i + 1) * 28] = nimg.reshape(28, 28)
        
        # Convert to tensor and ensure correct shape (batch_size, channels, height, width)
        nimg_tensor = torch.tensor(nimg, dtype=torch.float32).view(1, 1, 28, 28).to(device)
        
        with torch.no_grad():
            # Create dummy labels for the model
            dummy_labels = torch.zeros(1, 10, dtype=torch.float32).to(device)
            _, _, probs, _, _, _, _, _,_, _, _, _, uncertainty = model(nimg_tensor, dummy_labels)
            lu.append(uncertainty.mean().item())
            
        probs = probs.squeeze(0).cpu().numpy()
        scores += (probs >= prob_threshold).astype(float)
        ldeg.append(deg)
        lp.append(probs)
    
    # Convert lists to numpy arrays
    lp = np.array(lp)
    lu = np.array(lu)
    
    # Get labels that have any predictions above threshold
    labels = np.arange(10)[scores[0].astype(bool)]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot classification probabilities
    colors =['black','blue','red','brown','purple','cyan']
    markers = ['s','^','o']*2
    
    for i, label in enumerate(labels):
        ax1.plot(ldeg, lp[:, label], marker=markers[i], color=colors[i], label=f'Class {label}')
    
    # Plot uncertainty
    ax1.plot(ldeg, lu, marker='<', color='red', label='Uncertainty', linewidth=2)
    
    ax1.set_xlim([0, Mdeg])
    ax1.set_xlabel('Rotation Degree')
    ax1.set_ylabel('Probability')
    ax1.set_title('Classification Probabilities and Uncertainty vs Rotation')
    ax1.grid(True)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot rotated images
    ax2.imshow(1 - rimgs, cmap='gray')
    ax2.set_title('Rotated Images')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/Classification_Probability.png', bbox_inches='tight', dpi=300)
    plt.close()

def roc_test(model, testset, out_testset):
    """
    Calculate ROC curve for OOD detection using evidence values.
    Args:
        model: Trained UncertaintyLeNet model
        testset: In-distribution test dataset
        out_testset: Out-of-distribution test dataset
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Prepare in-distribution data
    normal_data = torch.stack([testset[i][0] for i in range(len(testset))]).to(device)
    normal_labels = torch.zeros(len(normal_data), 10, dtype=torch.float32).to(device)
    
    # Prepare out-of-distribution data
    anormal_data = torch.stack([out_testset[i][0] for i in range(len(out_testset))]).to(device)
    anormal_labels = torch.zeros(len(anormal_data), 10, dtype=torch.float32).to(device)
    
    # Get evidence values for both distributions
    with torch.no_grad():
        # In-distribution evidence
        _, _, _, _, _, _, _, evidence_normal, _, _, _, _, _ = model(normal_data, normal_labels)
        evidence_normal = evidence_normal.sum(dim=1).cpu().numpy()
        
        # Out-of-distribution evidence
        _, _, _, _, _, _, _, evidence_anormal, _, _, _, _, _ = model(anormal_data, anormal_labels)
        evidence_anormal = evidence_anormal.sum(dim=1).cpu().numpy()
    
    # Create labels (1 for in-distribution, 0 for out-of-distribution)
    y_true = np.concatenate([np.ones(len(evidence_normal)), np.zeros(len(evidence_anormal))])
    y_scores = np.concatenate([evidence_normal, evidence_anormal])
    
    # Calculate ROC curve
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('results/roc_curve_ood.png')
    plt.close()
    
    print(f'ROC AUC Score for OoD Detection: {roc_auc:.4f}')
    
    return roc_auc


def evaluate_uncertainty(model, test_loader, ood_loader=None):
    """
    Evaluate model uncertainty on in-distribution and out-of-distribution data.
    Args:
        model: Trained UncertaintyLeNet model
        test_loader: DataLoader for in-distribution test data
        ood_loader: DataLoader for out-of-distribution test data
    """
    device = next(model.parameters()).device
    model.eval()

    entropy_id = []  # Entropy for in-distribution samples
    entropy_id_correct = []  # Entropy for correctly classified in-distribution samples
    entropy_id_wrong = []  # Entropy for incorrectly classified in-distribution samples
    entropy_ood = []  # Entropy for out-of-distribution samples

    def calc_entropy(probs):
        # Ensure probabilities are valid (sum to 1 and non-negative)
        probs = F.softmax(probs, dim=1)
        return -torch.sum(probs * torch.log(probs + 1e-10), dim=1).cpu().numpy()
    
    # in distribution case
    with torch.inference_mode():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            Y = F.one_hot(targets, num_classes=10).float()
            
            _, _, probs, pred, truth, _, _, _, _, _, _, _, _ = model(data, Y)
            
            batch_entropy = calc_entropy(probs)
            entropy_id.extend(batch_entropy)
            
            for i, (p, t) in enumerate(zip(pred, truth)):
                if p == t:
                    entropy_id_correct.append(batch_entropy[i])
                else:
                    entropy_id_wrong.append(batch_entropy[i])
    
    # OOD case
    if ood_loader is not None:
        with torch.inference_mode():
            for data in ood_loader:
                if isinstance(data, (list, tuple)):
                    data = data[0]
                data = data.to(device)
                dummy_labels = torch.zeros(data.size(0), 10, dtype=torch.float32).to(device)
                
                _, _, probs, _, _, _, _, _, _, _, _, _, uncertainty = model(data, dummy_labels)
                batch_entropy = calc_entropy(probs)
                entropy_ood.extend(batch_entropy)
    
    # Convert to numpy arrays and ensure they're not empty
    entropy_id = np.array(entropy_id)
    entropy_id_correct = np.array(entropy_id_correct)
    entropy_id_wrong = np.array(entropy_id_wrong)
    entropy_ood = np.array(entropy_ood) if len(entropy_ood) > 0 else np.array([])
    
    # Print statistics for debugging
    print("\nEntropy Statistics:")
    print(f"In-Distribution Entropy - Mean: {np.mean(entropy_id):.4f}, Std: {np.std(entropy_id):.4f}")
    print(f"Correct Predictions - Mean: {np.mean(entropy_id_correct):.4f}, Std: {np.std(entropy_id_correct):.4f}")
    print(f"Wrong Predictions - Mean: {np.mean(entropy_id_wrong):.4f}, Std: {np.std(entropy_id_wrong):.4f}")
    if len(entropy_ood) > 0:
        print(f"Out-of-Distribution - Mean: {np.mean(entropy_ood):.4f}, Std: {np.std(entropy_ood):.4f}")
    
    # Plot 1: In-Distribution vs Out-of-Distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    if len(entropy_id) > 0:
        plt.hist(entropy_id, bins=50, alpha=0.5, label='In-Distribution (All)', density=True, color='blue')
    if len(entropy_ood) > 0:
        plt.hist(entropy_ood, bins=50, alpha=0.5, label='Out-of-Distribution', density=True, color='red')
    plt.xlabel('Entropy')
    plt.ylabel('Density')
    plt.title('Entropy Distribution for In-Distribution vs Out-of-Distribution')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Correct vs Wrong Predictions
    plt.subplot(2, 1, 2)
    if len(entropy_id_correct) > 0:
        plt.hist(entropy_id_correct, bins=50, alpha=0.5, label='Correct Predictions', density=True, color='green')
    if len(entropy_id_wrong) > 0:
        plt.hist(entropy_id_wrong, bins=50, alpha=0.5, label='Wrong Predictions', density=True, color='red')
    plt.xlabel('Entropy')
    plt.ylabel('Density')
    plt.title('Entropy Distribution for Correct vs Wrong Predictions')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/uncertainty_evaluation.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Plot 3: CDF of Entropy
    plt.figure(figsize=(10, 6))
    
    # Sort and calculate CDF for each distribution
    if len(entropy_id_correct) > 0:
        entropy_id_correct.sort()
        cdf_id_correct = np.arange(1, len(entropy_id_correct) + 1) / len(entropy_id_correct)
        plt.plot(entropy_id_correct, cdf_id_correct, label='ID (Correct)', color='green', linewidth=2)
    
    if len(entropy_id_wrong) > 0:
        entropy_id_wrong.sort()
        cdf_id_wrong = np.arange(1, len(entropy_id_wrong) + 1) / len(entropy_id_wrong)
        plt.plot(entropy_id_wrong, cdf_id_wrong, label='ID (Wrong)', color='red', linewidth=2)
    
    if len(entropy_ood) > 0:
        entropy_ood.sort()
        cdf_ood = np.arange(1, len(entropy_ood) + 1) / len(entropy_ood)
        plt.plot(entropy_ood, cdf_ood, label='OOD', color='blue', linewidth=2)
    
    plt.xlabel('Entropy')
    plt.ylabel('CDF')
    plt.title('Cumulative Distribution Function of Entropy')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('results/entropy_cdf.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Calculate and print ROC-AUC for OOD detection if OOD data is available
    if len(entropy_ood) > 0:
        from sklearn.metrics import roc_auc_score, roc_curve
        
        # Combine in-distribution and out-of-distribution entropies
        entropies = np.concatenate([entropy_id, entropy_ood])
        labels = np.concatenate([np.zeros(len(entropy_id)), np.ones(len(entropy_ood))])
        
        # Calculate ROC curve and AUC
        auc = roc_auc_score(labels, entropies)
        print(f"\nROC-AUC for OOD Detection: {auc:.4f}")
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(labels, entropies)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for OOD Detection')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('results/roc_curve.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()


def visualize_uncertainty_on_images(model, test_loader, num_samples=10):
    """
    Visualize uncertainty on images from the test set.
    Args:
        model: Trained UncertaintyLeNet model
        test_loader: DataLoader for test data
        num_samples: Number of samples to visualize
    """
    device = next(model.parameters()).device
    model.eval()

    data_samples = []
    target_samples = []
    batch_idx = 0

    for data, targets in test_loader:
        data_samples.append(data)
        target_samples.append(targets)
        batch_idx += 1
        if batch_idx == num_samples:
            break

    data_all = torch.cat(data_samples[:num_samples], dim=0)
    targets_all = torch.cat(target_samples[:num_samples], dim=0)

    with torch.inference_mode():
        data_all = data_all.to(device)
        targets_all = targets_all.to(device)
        Y = F.one_hot(targets_all, num_classes=10).float()

        _, _, probs, pred, truth, match, _, _, _, _, _, _, uncertainty = model(data_all, Y)

    # Select samples from both correct and wrong predictions
    indices = []
    correct_indices = (pred == truth).nonzero(as_tuple=True)[0].cpu()
    wrong_indices = (pred != truth).nonzero(as_tuple=True)[0].cpu()

    num_correct = min(num_samples // 2, len(correct_indices))
    num_wrong = min(num_samples - num_correct, len(wrong_indices))

    # In case there is no enough wrong predictions
    if num_wrong < num_samples - num_correct:
        num_correct = num_samples - num_wrong

    # Sort the cases by uncertainty
    correct_uncertainty = uncertainty.squeeze()[correct_indices].cpu()
    wrong_uncertainty = uncertainty.squeeze()[wrong_indices].cpu()

    sorted_correct = sorted(zip(correct_indices, correct_uncertainty), key=lambda x: x[1], reverse=True)
    sorted_wrong = sorted(zip(wrong_indices, wrong_uncertainty), key=lambda x: x[1], reverse=True)

    # Select uncertain samples
    if num_correct > 0 and len(sorted_correct) > 0:
        step = max(1, len(sorted_correct) // num_correct)
        indices.extend([sorted_correct[i][0] for i in range(0, len(sorted_correct), step)][:num_correct])

    if num_wrong > 0 and len(sorted_wrong) > 0:
        step = max(1, len(sorted_wrong) // num_wrong)
        indices.extend([sorted_wrong[i][0] for i in range(0, len(sorted_wrong), step)][:num_wrong])

    fig, axes = plt.subplots(nrows=len(indices), ncols=1, figsize=(6, 2*len(indices)))

    for i, idx in enumerate(indices):
        img = data_all[idx].cpu().squeeze().numpy()
        true_label = truth[idx].item()
        pred_label = pred[idx].item()
        unc_value = uncertainty[idx].item()

        title = f"True: {true_label}, Pred: {pred_label}, Uncertainty: {unc_value:.4f}"

        is_correct = true_label == pred_label
        border_color = 'green' if is_correct else 'red'

        if len(indices) == 1:
            ax = axes
        else:
            ax = axes[i]

        ax.imshow(img)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(2)

    # Just in case it looks ugly
    plt.tight_layout()
    plt.savefig('results/uncertainty_visualization.png')
    plt.close()

#----------------------------#
# Main Execution Function    #
#----------------------------#

def main(args):
    # Set up hyperparameters
    batch_size = args.batch_size
    vae_epochs = args.vae_epochs
    class_epochs = args.class_epochs

    # Data Loading
    print("Loading MNIST dataset...")
    trainset, testset, trainloader, testloader, out_trainset, out_testset, out_trainloader, out_testloader = load_mnist_data(batch_size=batch_size)

    # OOD Data Loading
    notmnist_data, notmnist_loader,OoD_dataset,OoD_loader = load_notmnist(batch_size=batch_size)

    model = UncertaintyLeNet(K=args.K, latent_dim=args.latent_dim).to(device)

    # Train VAE-GAN component
    vae_history = train_vae_gan(model.autoencoder, trainloader, num_epochs=args.vae_epochs)

    # Plot VAE-GAN training
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(vae_history['rec_loss'])
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(vae_history['disc_loss'])
    plt.title('Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig('results/vae_training_history.png')
    plt.close()

    # Train classifier
    model, history = train_model(model, trainloader, testloader, num_epochs=args.class_epochs)
    # Plot classifier training history
    plot_training_history(history)
    plot_original_and_generated_images(model, testset)


    # Evaluate uncertainty
    roc_test(model,testset, out_testset)
    rotated_image=trainset[6][0]
    rotating_image_classification(rotated_image, model, keep_prob=None, prob_threshold=0.5)
    evaluate_uncertainty(model, testloader, OoD_loader)

    # Visualize uncertainty on selected images
    visualize_uncertainty_on_images(model, testloader)

    print("Training and evaluation complete. Results saved to 'results' directory.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train and evaluate the uncertainty model')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--vae_epochs', type=int, default=50, help='Number of epochs for VAE training')
    parser.add_argument('--class_epochs', type=int, default=50, help='Number of epochs for classifier training')
    parser.add_argument('--K', type=int, default=10, help='Number of classes')
    parser.add_argument('--latent_dim', type=int, default=100, help='Latent dimension')
    args = parser.parse_args()
    main(args=args)
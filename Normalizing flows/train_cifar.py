import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from net_work import discretize, show_imgs, create_cifar_multiscale_flow

DATASET_PATH = "./dataset"
CHECKPOINT_PATH = "./saved_model_ckpt/"

pl.seed_everything(42)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

transform = transforms.Compose([transforms.ToTensor(), discretize])
train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=transform, download=True)
pl.seed_everything(42)
train_set, val_set = torch.utils.data.random_split(train_dataset, [45000, 5000])
test_set = CIFAR10(root=DATASET_PATH, train=False, transform=transform, download=True)


train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=False)
val_loader = data.DataLoader(val_set, batch_size=64, shuffle=False, drop_last=False, num_workers=4)
test_loader = data.DataLoader(test_set, batch_size=64, shuffle=False, drop_last=False, num_workers=4)

# Create a PyTorch Lightning trainer
model_name = "CIFAR_Flow"
flow = create_cifar_multiscale_flow()
trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, model_name),
                        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                        devices=1,
                        max_epochs=300,
                        gradient_clip_val=1.0,
                        callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_bpd"),
                                LearningRateMonitor("epoch")],
                        check_val_every_n_epoch=30)

trainer.logger._log_graph = True
trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
print("Start training", model_name)
trainer.fit(flow, train_loader, val_loader)
val_result = trainer.test(flow, val_loader, verbose=False)
start_time = time.time()
test_result = trainer.test(flow, test_loader, verbose=False)
duration = time.time() - start_time
result = {"test": test_result, "val": val_result, "time": duration / len(test_loader) / flow.import_samples}
torch.save(flow.state_dict(), "./saved_model_ckpt/cifar_flow.pth")
print("Finish train cifar flow")




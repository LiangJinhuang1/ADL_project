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
from torchvision import datasets, transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from net_work import discretize, show_imgs, create_lsun_multiscale_flow, LSUN_IMG_SZ

DATASET_PATH = "./dataset"
CHECKPOINT_PATH = "./saved_model_ckpt/"

pl.seed_everything(42)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
print("Using device", device)

transform = transforms.Compose([transforms.Resize((LSUN_IMG_SZ, LSUN_IMG_SZ)),
                                transforms.ToTensor(), discretize])


dataset = datasets.LSUN(DATASET_PATH + '/LSUN', classes=["bedroom_val"], transform=transform)
pl.seed_everything(42)
train_set, val_set = torch.utils.data.random_split(dataset, [240, 60])
test_set = val_set

train_loader = data.DataLoader(train_set, batch_size=8, shuffle=True, drop_last=False)
val_loader = data.DataLoader(val_set, batch_size=4, shuffle=False, drop_last=False, num_workers=4)
test_loader = data.DataLoader(test_set, batch_size=4, shuffle=False, drop_last=False, num_workers=4)

# Create a PyTorch Lightning trainer
model_name = "LSUN_Flow"
flow = create_lsun_multiscale_flow()
trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, model_name),
                        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                        devices=1,
                        max_epochs=4,
                        gradient_clip_val=1.0,
                        callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_bpd"),
                                LearningRateMonitor("epoch")],
                        check_val_every_n_epoch=3)

trainer.logger._log_graph = True
trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
print("Start training", model_name)
trainer.fit(flow, train_loader, val_loader)
val_result = trainer.test(flow, val_loader, verbose=False)
start_time = time.time()
test_result = trainer.test(flow, test_loader, verbose=False)
duration = time.time() - start_time
result = {"test": test_result, "val": val_result, "time": duration / len(test_loader) / flow.import_samples}
torch.save(flow.state_dict(), "./saved_model_ckpt/LSUN_flow.pth")
print("Finish train LSUN flow")




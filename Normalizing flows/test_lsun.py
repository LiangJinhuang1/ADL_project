import torch.nn as nn
import torch
import pytorch_lightning as pl
import torchvision
from torchvision import transforms
from net_work import show_imgs, create_lsun_multiscale_flow, show_imgs_save

DATASET_PATH = "./dataset"
CHECKPOINT_PATH = "./saved_model_ckpt/"

pl.seed_everything(42)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

pretrained_filename =  "./saved_model_ckpt/LSUN_flow.pth"
ckpt = torch.load(pretrained_filename)
flow = create_lsun_multiscale_flow()
flow.load_state_dict(ckpt)

samples = flow.sample(img_shape=[4,24,28, 28])
show_imgs_save(samples, image_path="./result_visualization/LSUN_flow.png", row_size=2)
print("finish test")
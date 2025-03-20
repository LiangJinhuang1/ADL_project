import torch.nn as nn
import torch
import pytorch_lightning as pl
import torchvision
from torchvision import transforms
from net_work import show_imgs, create_cifar_multiscale_flow, show_imgs_save

DATASET_PATH = "./dataset"
CHECKPOINT_PATH = "./saved_model_ckpt/"

pl.seed_everything(42)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

pretrained_filename =  "./saved_model_ckpt/cifar_flow.pth"
ckpt = torch.load(pretrained_filename)
flow = create_cifar_multiscale_flow()
flow.load_state_dict(ckpt)

samples = flow.sample(img_shape=[32,24,8,8])
show_imgs_save(samples, image_path="./result_visualization/cifar_flow2.png")
print("finish test")
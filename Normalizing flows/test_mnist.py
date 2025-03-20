import torch.nn as nn
import torch
import pytorch_lightning as pl
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from net_work import show_imgs, create_mnist_multiscale_flow, show_imgs_save

DATASET_PATH = "./dataset"
CHECKPOINT_PATH = "./saved_model_ckpt/"

pl.seed_everything(42)
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

pretrained_filename =  "./saved_model_ckpt/mnist_flow.pth"
# if use CPU
ckpt = torch.load(pretrained_filename, map_location=torch.device('cpu') )
# if use GPU
# ckpt = torch.load(pretrained_filename)
flow = create_mnist_multiscale_flow()
flow.load_state_dict(ckpt)

samples = flow.sample(img_shape=[96,8,7,7])
show_imgs_save(samples, image_path="./result_visualization/mnist_flow.png")
print("finish test")
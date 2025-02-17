from json import load
import os
import argparse
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sympy import im
import torch
from torch import nn
import torchvision
from torchvision import transforms
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from einops import reduce, rearrange

from utils.data_utils_acc import get_loader

from models.vringformer import VRingFormer, VRingFormer_CONFIGS
from models.vanilla_transformer import VisionTransformer, ViT_CONFIGS
from models.one_wide_feed_forward import OWF, OWF_CONFIGS
from models.universal_transformer import UiT, UiT_CONFIGS

def get_transform(args):
    
    if args.dataset == "cifar10" or args.dataset == "cifar100":
        # Define transformations for CIFAR
        cifar_transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),        
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), 
        ])
        return cifar_transform_test
    
    elif args.dataset == "imagenet100":
        # Define transformations for ImageNet100    
        imagenet100_transform_test = transforms.Compose([
        transforms.Resize(256),                         # Resize image to 256x256
        transforms.CenterCrop(224),                     # Center crop image to 224x224
        transforms.ToTensor(),                          # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet statistics
        ])
        return imagenet100_transform_test  
    elif args.dataset == "tiny_imagenet200":
        # Define transformations for ImageNet200
        imagenet200_transform_test = transforms.Compose([
        transforms.Resize(256),                         # Resize image to 256x256
        transforms.CenterCrop(224),                     # Center crop image to 224x224
        transforms.ToTensor(),                          # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),  # Normalize with ImageNet statistics
        ])
        return imagenet200_transform_test
    elif args.dataset == "imagenet1k":
        # Define transformations for ImageNet1k
        imagenet1k_transform_test = transforms.Compose([
        transforms.Resize(256),                         # Resize image to 256x256
        transforms.CenterCrop(224),                     # Center crop image to 224x224
        transforms.ToTensor(),                          # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet statistics
        ])
        return imagenet1k_transform_test
    else:
        raise NotImplementedError
    

def load_model(args, model):
    model_dir = args.pretrained_dir
    if os.path.exists(model_dir):
        model_info = torch.load(model_dir)
        if "model_state_dict" in model_info.keys():
            model.load_state_dict(model_info["model_state_dict"])
        else:
            model.load_state_dict(model_info) # no other info saved
            
        print("Loaded model checkpoint from %s" % model_dir)
    else:
        print("No checkpoint found in [DIR: %s]", model_dir)
    
    return model

def setup_model(args):

    if args.dataset == "cifar10":
        num_classes = 10
    elif args.dataset == "cifar100":
        num_classes = 100
    elif args.dataset == "imagenet1k":
        num_classes = 1000
    elif args.dataset == "imagenet100":
        num_classes = 100
    elif args.dataset == "tiny_imagenet200":
        num_classes = 200
    else:
        raise NotImplementedError

    if args.model_type.lower() == "vringformer":
        config = VRingFormer_CONFIGS[args.model_size]
        model_type = VRingFormer
    elif args.model_type.lower() == "vit":
        config = ViT_CONFIGS[args.model_size]
        model_type = VisionTransformer
    elif args.model_type.lower() == "owf":
        config = OWF_CONFIGS[args.model_size]
        model_type = OWF
    elif args.model_type.lower() == "uit":
        config = UiT_CONFIGS[args.model_size]
        model_type = UiT
    else:
        raise NotImplementedError
    
    model = model_type(config, args.img_size, num_classes=num_classes)
    
    model = load_model(args, model)
    model.eval() # Set model to evaluation mode
    model.to(args.device) # Move model to device
    
    return model, config

def vis_attention_map_high_res(attentions, image, num_of_heads, patch_size, output_dir):
  
  att_mat = torch.stack(attentions).squeeze(1)

  att_mat = reduce(att_mat, 'b h len1 len2 -> b len1 len2', 'mean')
  im = image # np.array(image)

  residual_att = torch.eye(att_mat.size(1))
  aug_att_mat = att_mat + residual_att
  aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

  # Recursively multiply the weight matrices
  joint_attentions = torch.zeros(aug_att_mat.size())
  joint_attentions[0] = aug_att_mat[0]

  for n in range(1, aug_att_mat.size(0)):
      joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
      
  # Attention from the output token to the input space.
  v = joint_attentions[-1]
  grid_size = int(np.sqrt(aug_att_mat.size(-1)))
  mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
  mask = cv2.resize(mask / mask.max(), (im.shape[1], im.shape[0]))[..., np.newaxis]
  result = (mask * im).astype("uint8")

  fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(16, 16))

  ax1.set_title('Original')
  ax2.set_title('Attention Mask')
  ax3.set_title('Attention Map')
  _ = ax1.imshow(im)
  _ = ax2.imshow(mask.squeeze())
  _ = ax3.imshow(result)


def vis_attention_map(attentions, image, num_of_heads, patch_size, output_dir):
    threshold = 0.6
    w_featmap = image.shape[-2] // patch_size
    h_featmap = image.shape[-1] // patch_size

    # we keep only a certain percentage of the mass
    val, idx = torch.sort(attentions)
    val /= torch.sum(val, dim=1, keepdim=True)
    cumval = torch.cumsum(val, dim=1)
    th_attn = cumval > (1 - threshold)
    idx2 = torch.argsort(idx)
    for head in range(num_of_heads):
        th_attn[head] = th_attn[head][idx2[head]]
    th_attn = th_attn.reshape(num_of_heads, w_featmap, h_featmap).float()
    # interpolate
    
    th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

    attentions = attentions.reshape(num_of_heads, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu()
    attentions = attentions.detach().numpy()

    # show and save attentions heatmaps
    
    os.makedirs(output_dir, exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(image, normalize=True, scale_each=True), os.path.join(output_dir, "img.png"))
    for j in range(num_of_heads):
        fname = os.path.join(output_dir, "attn-head" + str(j) + ".png")
        plt.figure()
        plt.imshow(attentions[j])
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        #print(f"{fname} saved.")
        # plt.close()
    exit()


def main():

    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--model_size", type=str, required=True)
    parser.add_argument("--pretrained_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    seed = args.seed
    
    torch.manual_seed(seed) # set seed for reproducibility
    np.random.seed(seed) # set seed for reproducibility
    
    model, config = setup_model(args)
    patch_size = config.patches['size'][0]
    num_of_images = args.num_samples
    
    output_dir= args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # randomly sample images a certain number of from the dataset consisting of class folders
    # image_paths = []
    # for root, dirs, files in os.walk(args.data_dir):
    #     for file in files:
    #         if file.endswith(".JPEG") or file.endswith(".jpg") or file.endswith(".png"):
    #             image_paths.append(os.path.relpath(os.path.join(root, file), args.data_dir))
    
    # print(f"Number of images in the dataset: {len(image_paths)}")
    
    # image_paths = np.random.choice(image_paths, num_of_images, replace=False)    
    
    # img_transform = get_transform(args)
    # image = Image.open(os.path.join(args.data_dir, image_paths[0])).convert("RGB")
    # print(img_transform(image).shape)
    
    _, test_loader = get_loader(args)
    
    for i, (images, label) in enumerate(test_loader):  
        
        with torch.no_grad():
            images = images.to(device)
            label = label.to(device)
            pred, attention_scores = model(images)
            # get the index of the images that are correctly classified
            pred = pred.argmax(dim=1)
            correct_idx = (pred == label).nonzero().squeeze().cpu()
            if correct_idx.numel() == 0 or correct_idx.numel() < num_of_images:
                continue
            
            chosen_images_idx = np.random.choice(correct_idx, num_of_images, replace=False) 
    
            attention_scores = attention_scores[-1] # get the last layer attention scores
            
            for idx in chosen_images_idx:
        
                image = images[idx]
                attention_info = attention_scores[idx]
                num_of_heads = attention_info.shape[0]
                cls_token_attn = attention_info[:, 0, 1:].reshape(num_of_heads, -1)
                
                # print(f"Image shape: {image.shape}")
                # print(f"Attention scores shape: {attention_scores.shape}")
                # print(f"CLS token attention shape: {cls_token_attn.shape}")

                vis_attention_map(cls_token_attn, image, num_of_heads, patch_size, output_dir)            
        
        break
    
    
if __name__ == "__main__":
    main()
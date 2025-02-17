from json import load
import os
import argparse
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sympy import im
import torch
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

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
            model_info["model_state_dict"] = {re.sub(r"_orig_mod.", "", k): v for k, v in model_info["model_state_dict"].items()}
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
    model.eval()
    model.to(args.device) # Move model to device
    
    return model, config


def compute_distance_matrix(patch_size, num_patches, length):
    distance_matrix = np.zeros((num_patches, num_patches))
    for i in range(num_patches):
        for j in range(num_patches):
            if i == j:  # zero distance
                continue

            xi, yi = (int(i / length)), (i % length)
            xj, yj = (int(j / length)), (j % length)
            distance_matrix[i, j] = patch_size * np.linalg.norm([xi - xj, yi - yj])

    return distance_matrix

def compute_mean_attention_dist(patch_size, attention_weights):
    num_cls_tokens = 1
    
    # The attention_weights shape = (batch, num_heads, num_patches, num_patches)
    attention_weights = attention_weights[
        ..., num_cls_tokens:, num_cls_tokens:
    ]  # Removing the CLS token
    num_patches = attention_weights.shape[-1]
    length = int(np.sqrt(num_patches))
    assert length**2 == num_patches, "Num patches is not perfect square"

    distance_matrix = compute_distance_matrix(patch_size, num_patches, length)
    h, w = distance_matrix.shape

    distance_matrix = distance_matrix.reshape((1, 1, h, w))
    # The attention_weights along the last axis adds to 1
    # this is due to the fact that they are softmax of the raw logits
    # summation of the (attention_weights * distance_matrix)
    # should result in an average distance per token.
    mean_distances = attention_weights * distance_matrix
    mean_distances = np.sum(
        mean_distances, axis=-1
    )  # Sum along last axis to get average distance per token
    mean_distances = np.mean(
        mean_distances, axis=-1
    )  # Now average across all the tokens
    
    return mean_distances


def main():

    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--model_size", type=str, required=True)
    parser.add_argument("--pretrained_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--extra_info", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)


    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    dataset = args.dataset
    seed = args.seed
    
    torch.manual_seed(seed) # set seed for reproducibility
    np.random.seed(seed) # set seed for reproducibility
    
    model, config = setup_model(args)
    
    model_type = args.model_type
    model_size = args.model_size
    extra_info = args.extra_info
    patch_size = config.patches['size'][0]
    num_of_images = args.num_samples
    
    output_dir= args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # randomly sample images a certain number of from the dataset consisting of class folders
    image_paths = []
    for root, dirs, files in os.walk(args.data_dir):
        for file in files:
            if file.endswith(".JPEG") or file.endswith(".jpg") or file.endswith(".png"):
                image_paths.append(os.path.relpath(os.path.join(root, file), args.data_dir))
    
    print(f"Number of images in the dataset: {len(image_paths)}")
    
    image_paths = np.random.choice(image_paths, num_of_images, replace=False)    
    
    img_transform = get_transform(args)
    # image = Image.open(os.path.join(args.data_dir, image_paths[0])).convert("RGB")
    # print(img_transform(image).shape)
    
    all_mean_distances = {}
    for img_idx, image_path in tqdm(enumerate(image_paths)):
        image = Image.open(os.path.join(args.data_dir, image_path)).convert("RGB")
        image = img_transform(image).unsqueeze(0)
        
        with torch.no_grad():
            image = image.to(device)
            pred, attention_scores = model(image)
        
        attention_scores = [attention.cpu().numpy() for attention in attention_scores]
        
        for block_idx, attention_score_matrix in enumerate(attention_scores):
            mean_distance = compute_mean_attention_dist(
                patch_size=patch_size,
                attention_weights=attention_score_matrix
            )
            
            if img_idx == 0:
                all_mean_distances[f"transformer_block_{block_idx}_attn_mean_dist"] = mean_distance
            else: # concatenate the mean distances
                all_mean_distances[f"transformer_block_{block_idx}_attn_mean_dist"] = np.concatenate(
                    (all_mean_distances[f"transformer_block_{block_idx}_attn_mean_dist"], mean_distance), axis=0
                )
    
    # numner of attention heads
    num_heads = all_mean_distances[f"transformer_block_0_attn_mean_dist"].shape[-1]
    
    print(f"Num Heads: {num_heads}")
    
    # average the mean distances
    for key, value in all_mean_distances.items():
        all_mean_distances[key] = np.mean(value, axis=0, keepdims=True)
    
    plt.figure(figsize=(8, 8))

    for idx in range(len(all_mean_distances)):
        mean_distance = all_mean_distances[f"transformer_block_{idx}_attn_mean_dist"]
        x = [idx + 1] * num_heads
        y = mean_distance[0, :]
        # enlarge the size of the points
        plt.scatter(x=x, y=y, s=60)
        # plt.scatter(x=x, y=y) # label=f"transformer_block_{idx}"

    # plt.legend(loc="lower right", fontsize=10)
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.xticks(range(1, len(all_mean_distances) + 1)) # transformer layers
    plt.xlabel("Transformer Layers", fontsize=18)
    plt.ylabel("Mean Attention Distance", fontsize=18)
    plt.title(f"MAD of {model_type} Attention Heads", fontsize=18)
    plt.grid(alpha=0.5)
    name = model_size + "_" + extra_info if extra_info else model_size
    plt.savefig(os.path.join(output_dir, f"{dataset}_mean_attention_distance_{name}.png"))

if __name__ == "__main__":
    main()
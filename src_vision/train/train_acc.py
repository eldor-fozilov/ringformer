import logging
import argparse
import os
import random
import re
import time
import numpy as np
import sys

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from accelerate import Accelerator

from models.vringformer import VRingFormer, VRingFormer_CONFIGS
from models.vanilla_transformer import VisionTransformer, ViT_CONFIGS
from models.one_wide_feed_forward import OWF, OWF_CONFIGS
from models.universal_transformer import UiT, UiT_CONFIGS

from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils_acc import get_loader
from utils.dist_util import get_rank, get_world_size

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def save_model(args, epoch, model, optimizer, scheduler, best_acc, global_step):
    save_dir = os.path.join(args.output_dir, args.model_type.lower())
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(save_dir, "%s_checkpoint.bin" % args.name)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_acc': best_acc,
        'global_step': global_step
        
    }
    
    torch.save(checkpoint, model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", save_dir)

def load_model(args, model, optimizer, scheduler):
    model_dir = os.path.join(args.output_dir, args.model_type.lower(), "%s_checkpoint.bin" % args.name)
    
    if os.path.exists(model_dir):
        model_info = torch.load(model_dir)
        # the keys in the model state dict are prefixed with "_orig_mod."
        # so we need to remove the prefix
        model_info["model_state_dict"] = {re.sub(r"_orig_mod.", "", k): v for k, v in model_info["model_state_dict"].items()}
    
        
        model.load_state_dict(model_info["model_state_dict"])
        optimizer.load_state_dict(model_info["optimizer_state_dict"])
        scheduler.load_state_dict(model_info["scheduler_state_dict"])
        best_acc = model_info["best_acc"]
        epoch = model_info["epoch"]
        global_step = model_info["global_step"]
        logger.info("Loaded model checkpoint and training state related info (optimizer, scheduler, best_acc, global_step) from [DIR: %s]", model_dir)
    else:
        logger.info("No checkpoint found in [DIR: %s]", model_dir)
    
    return model, optimizer, scheduler, best_acc, epoch, global_step

def setup(args):
    # Prepare model

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
    
    # if args.pretrained_dir is not None:
    #     model.load_from(args.pretrained_dir)
    #     print("Loaded pretrained model from %s" % args.pretrained_dir)
    
    # if not args.distributed_training: # one GPU
    #     model.to(args.device)
    
    num_params = count_parameters(model)

    if args.local_rank in [-1, 0]:
        logger.info("----------------- Model Configuration -----------------")
        logger.info("{}".format(config))
        logger.info("-----------------Training Arguments-----------------")
        logger.info("{}".format(args))
        logger.info("-----------------Model Size-----------------")
        logger.info(f"Total Parameter: {num_params:.2f}M")
        logger.info("---------------------------------------------------")
    
    args.model_config = config
    
    return args, model

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()


    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          total=len(test_loader),
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        x, y = batch
        
        if not args.distributed_training: # one GPU
            x, y = x.to(args.device), y.to(args.device)
        
        with torch.no_grad():
            logits = model(x)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy

def get_optimizer(model, args):

    print("CUSTOM OPTIMIZER USED")

    recurrsive_param = []
    other_param = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue    
        
        if args.model_type.lower() == "vringformer":
            if "attn" in name or 'ffn' in name:
                recurrsive_param.append(param)
            else:
                other_param.append(param)
        
        elif args.model_type.lower() == "uit" or args.model_type.lower() == "vit":
            pass
        
        elif args.model_type.lower() == "owf":
            if 'ff_block' in name:
                recurrsive_param.append(param)
            else:
                other_param.append(param)
        
        else:
            return NotImplementedError
    
    optimizer = None
    
    if args.model_type.lower() in ["vit", 'uit']:            
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.model_type.lower() == "vringformer":
        optimizer = torch.optim.Adam([{'params': recurrsive_param, 'lr': args.learning_rate * 2 / args.model_config.transformer["num_levels"]},
                                      {'params': other_param, 'lr': args.learning_rate}])
    elif args.model_type.lower() == "owf":
        optimizer = torch.optim.Adam([{'params': recurrsive_param, 'lr': args.learning_rate * 2 / args.model_config.transformer["num_layers"]}, {'params': other_param, 'lr': args.learning_rate}])
    else:
        return NotImplementedError
    
    return optimizer

def train(args, model):
    """ Train the model """
    
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("tensorboard_logs", args.name))

    args.train_batch_size = args.train_batch_size // (args.gradient_accumulation_steps * (args.world_size if args.local_rank != -1 else 1))

    # Prepare dataset
    train_loader, test_loader = get_loader(args)
    
    # Prepare optimizer and scheduler
    
    # optimizer = get_optimizer(model, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    if args.training_steps != -1:
        t_total = args.training_steps
    else:
        t_total = args.num_epochs * len(train_loader) // args.gradient_accumulation_steps
    
    if args.warmup_steps != -1:
        warmup_steps = args.warmup_steps
    else:
        warmup_steps = args.warmup_epochs * len(train_loader) // args.gradient_accumulation_steps 
    
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

    if args.pretrained_dir is not None:
        model, optimizer, scheduler, prev_best_acc, prev_epoch, prev_global_step = load_model(args, model, optimizer, scheduler)
    else:
        prev_best_acc, prev_epoch, prev_global_step = 0, 0, 0

    # Compile the model
    model = torch.compile(model)
        
    if args.distributed_training:
        # Distributed training using accelerator
        accelerator = Accelerator()
        args.device = accelerator.device
        train_loader, test_loader, model, optimizer, scheduler = accelerator.prepare(
            train_loader, test_loader, model, optimizer, scheduler
        )
    else:
        model.to(args.device) # one GPU

    # Train!
    if args.local_rank in [-1, 0]:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_loader.dataset))
        
        if args.num_epochs != -1:
            logger.info("  Num Epochs = %d", args.num_epochs)
        logger.info("  Warmup Steps = %d", warmup_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    args.train_batch_size * args.gradient_accumulation_steps * (
                        args.world_size if args.local_rank != -1 else 1))
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    
    early_stop = 0
    best_acc = prev_best_acc # if loaded from checkpoint
    global_step = prev_global_step # if loaded from checkpoint
    num_of_epochs = prev_epoch # if loaded from checkpoint
    
    eval_steps = args.eval_every * len(train_loader) // args.gradient_accumulation_steps
    total_start_time = time.time()
    while True:
        
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        
        if args.local_rank in [-1, 0]:
            logger.info("Epoch: {}" .format(num_of_epochs + 1))
       
        start_time = time.time()
        train_acc = 0
        for step, batch in enumerate(epoch_iterator):
            x, y = batch
            
            if not args.distributed_training: # one GPU
                x, y = x.to(args.device), y.to(args.device)
            
            logits, loss = model(x, y)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            with torch.no_grad():
                preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                train_acc += simple_accuracy(preds, y)

            if args.distributed_training:
                accelerator.backward(loss)
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item() * args.gradient_accumulation_steps)

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_last_lr()[0], global_step=global_step)
                if global_step % eval_steps == 0 and args.local_rank in [-1, 0]:
                    accuracy = valid(args, model, writer, test_loader, global_step)
                    early_stop += 1
                    if best_acc < accuracy:
                        save_model(args, num_of_epochs, model, optimizer, scheduler, accuracy, global_step)
                        best_acc = accuracy
                        early_stop = 0
            
                    model.train()
        
                if global_step % t_total == 0:
                    break
        
        losses.reset()
        end_time = time.time()
        
        if args.local_rank in [-1, 0]:
            logger.info("Epoch Time: %2.5f minutes" % ((end_time - start_time) / 60)) # in minutes
            logger.info("Train Accuracy: %2.5f" % (train_acc / len(train_loader)))
            
        if early_stop >= args.early_stop_criterion:
            logger.info("-" * 100)
            logger.warning("Early Stopping!")
            logger.info("-" * 100)
            break
        
        if global_step % t_total == 0:
            break
        
        num_of_epochs += 1
    
    total_end_time = time.time()

    if args.local_rank in [-1, 0]:
        writer.close()
        logger.info("-" * 100)
        logger.info("Total Training Time: %2.5f minutes" % ((total_end_time - total_start_time) / 60)) # in minutes
        logger.info("-" * 100)
        logger.info("Best Accuracy: \t%f" % best_acc)
        logger.info("End Training!")

def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--distributed_training", default=False, type=bool,
                        help="Whether to use distributed training")
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", required=True, default="cifar10")
    parser.add_argument("--model_type", required=True)
    parser.add_argument("--model_size", required=True, type=str,
                        help="Model size")
    parser.add_argument("--pretrained_dir", default=None)
    parser.add_argument("--output_dir", default="checkpoints", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=10, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--num_epochs", default=None, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--training_steps", default=-1, type=int, help="Total number of training steps to perform.")
    parser.add_argument("--early_stop_criterion", default=10, type=int)
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_epochs", default=None, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--warmup_steps", default=-1, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', type=bool, default=False,
                        help="Whether to use 16-bit float precision instead of 32-bit")
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    
    if args.distributed_training:    
        args.global_rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])

        device = torch.device("cuda", args.local_rank)
    else:  
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.local_rank = -1
    
    args.n_gpu = 1
    args.device = device

    # Setup logging
    log_dir = os.path.join("experiment_logs", args.model_type.lower())
    os.makedirs(log_dir, exist_ok=True) 
    
    logging.basicConfig(filename=os.path.join(log_dir, f"{args.name}.log"),
                        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.info("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.local_rank in [-1, 0]:
        print("-" * 100)
        print("Training/evaluation parameters")
        print("-" * 100)
        print(args)

    # Set seed
    set_seed(args)

    # Model Setup
    args, model = setup(args)

    # Training
    train(args, model)

if __name__ == "__main__":
    main()
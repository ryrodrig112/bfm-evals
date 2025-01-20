#!/usr/bin/env python3

from batcher.base import EEGDataset
from encoder.conformer_braindecode import EEGConformer
from embedder.base import EmbeddingModel, BaseEmbedder
from decoder.gpt import GPTModel, PretrainedGPT2
from decoder.unembedder import DeconvNet, UnEmbedder
from torch.utils.data import DataLoader
import pathlib
import zlib
import gzip
import torch
from model import Model
import yaml
import argparse
from argparse import Namespace
from typing import Dict, Any
import datetime, os
import torch
from torch.utils.tensorboard import SummaryWriter



def load_config(config_path: str)-> dict:
    with open(config_path, "r") as f:
        defaults = yaml.safe_load(f)
    return defaults

def get_config_args()->Namespace:
    parser = argparse.ArgumentParser(description="Set path to config file")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration YAML file")
    args, unknown = parser.parse_known_args()
    return args, unknown

def get_args(defaults: Dict[str, Any], remaining_args: list) -> Namespace:
    parser = argparse.ArgumentParser(description="Train EEG Model")

    # Data arguments
    parser.add_argument("--train_dir", type=str, default=defaults["data"]["train_dir"], help="Path to training data directory")
    parser.add_argument("--val_dir", type=str, default=defaults["data"]["val_dir"], help="Path to validation data directory")
    parser.add_argument("--batch_size", type=int, default=defaults["data"]["batch_size"], help="Batch size for DataLoader")
    parser.add_argument("--num_workers", type=int, default=defaults["data"]["num_workers"],
                        help="Number of workers for data loader")
    parser.add_argument("--shuffle", action="store_true", default=defaults["data"]["shuffle"], help="Shuffle the dataset")

    # Model arguments
    parser.add_argument("--encoder_n_chans", type=int, default=defaults['model']['encoder']['n_chans'],
                        help="Number of encoder channels")
    parser.add_argument("--encoder_n_times", type=int, default=defaults['model']['encoder']['n_times'],
                        help="Length of encoder sequence")
    parser.add_argument("--embedder_in_dim", type=int, default=defaults['model']['embedder']['in_dim'],
                        help="Dimension of embedder input")
    parser.add_argument("--unembedder_out_dim", type=int, default=defaults['model']['unembedder']['out_dim'],
                        help="Dimension of unembedder output")

    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=defaults["training"]["num_epochs"], help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=defaults["training"]["learning_rate"], help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=defaults["training"]["weight_decay"], help="L2 regularization")
    parser.add_argument("--gradient_clipping", type=float, default=None, help="Clip gradients to value if set")
    parser.add_argument("--smoketest", type=bool, default=defaults['training']['smoketest'], help="Set to true if its a test run")

    # Logging arguments
    parser.add_argument("--log_interval", type=int, default=defaults["logging"]["log_interval"],
                        help="Interval for logging progress")
    parser.add_argument("--save_model", action="store_true", default=defaults["logging"]["save_model"],
                        help="Whether to save the model")
    parser.add_argument("--tb_dir", type=str, default=defaults['logging']['tb_dir'], help="Specify path for tensorboard tracking")
    parser.add_argument("--model_checkpt_dir", type=str, default=defaults["logging"]["model_checkpt_dir"],
                        help="Directory to save model checkpoints")

    # Loading args
    parser.add_argument("--from_scratch", type=bool, default=defaults["loading"]["from_scratch"], help= "Specify whether model is loading from scratch or pretrained")
    parser.add_argument("--pretrained_checkpoint_dir", type=str, default=defaults["loading"]["pretrained_checkpoint_dir"], help= "path to existing model checkpoint")
    parser.add_argument("--pretrained_model_path", type=str, default=defaults["loading"]["pretrained_model_path"], help= "path to existing model checkpoint")
    parser.add_argument("--start_at_epoch", type=int, default=defaults["loading"]["start_at_epoch"], help= "epoch to resume training from")

    return parser.parse_args(remaining_args)


def load_data(data_dir: str, batch_size: int, shuffle: bool, num_workers: int, smoketest=False)->torch.utils.data.DataLoader:
    data_dir = pathlib.Path(data_dir)
    files = list(data_dir.glob("*.gz"))
    if smoketest:
        files = files[:10]

    data = EEGDataset(files)
    dataloader = DataLoader(data,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers)
    return dataloader

def construct_model(args):
    model = Model(
            encoder=EEGConformer(n_chans=args.encoder_n_chans, n_times=args.encoder_n_times),
            embedder=BaseEmbedder(in_dim=args.embedder_in_dim),
            decoder=PretrainedGPT2(),
            unembedder=UnEmbedder(out_dim=args.unembedder_out_dim))

    if args.from_scratch == False:
        print(f"Loading model from {args.pretrained_model_path}")
        model.from_pretrained(args.pretrained_model_path)
    return model

def train_one_epoch(model, training_dataloader, optimizer, args):
    model.train()
    total_loss = 0
    for step, batch in enumerate(training_dataloader):
        # Forward and backward pass
        print(step+1)
        losses, outputs = model.compute_loss(batch, return_outputs=True)
        loss = losses['loss']
        total_loss += loss.item()
        loss.backward()

        # Gradient clipping
        if args.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)

        optimizer.step()
        optimizer.zero_grad()
        # Logging
        # if (step + 1) % args.log_interval == 0:
        #     print(f"Step {step}, Loss: {loss.item()}")
    avg_batch_loss = total_loss / len(training_dataloader)
    return avg_batch_loss

def eval_model(model, val_dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            losses, outputs = model.compute_loss(batch, return_outputs=True)
            loss = losses['loss']
            total_loss += loss.item()
    avg_loss = total_loss / len(val_dataloader)
    return avg_loss

def save_model(model, output_dir: str, epoch):
    """Save the trained model to the specified directory."""
    print("Saving Model")

    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path / f"model_checkpoint_epoch{epoch}.pth")
    print("Model saved successfully!")

def train_model(model, train_dataloader, val_dataloader, optimizer, writer, args):
    if args.from_scratch == True:
        model_save_dir = os.path.join(args.model_checkpt_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        start = 0
    else:
        model_save_dir = os.path.join(args.model_checkpt_dir, args.pretrained_checkpoint_dir)
        start = args.start_at_epoch
    for epoch in range(args.num_epochs):
        print(f'Beginning epoch {epoch + 1 + start}')
        avg_training_loss = train_one_epoch(model, train_dataloader, optimizer, args)
        avg_val_loss = eval_model(model, val_dataloader)
        writer.add_scalars("Loss",{
            'Training Loss': avg_training_loss,
            'Validation Loss': avg_val_loss}, epoch)
        print(f"Avg Batch Training Loss: {avg_training_loss}")
        print(f"Avg Batch Val Loss: {avg_val_loss}")
        if args.from_scratch==True:
            save_model(model, model_save_dir, epoch+1)
        else:
            save_model(model, model_save_dir, epoch+args.start_at_epoch+1)
        print("------------")


# config_path="config.yaml"


if __name__ == "__main__":
    # Load defaults configuration and overwrite w/ command line args passed in
    print("Fetching Configuration")
    config_args, remaining_args = get_config_args()
    defaults = load_config(config_args.config_path)
    args = get_args(defaults, remaining_args)
    writer = SummaryWriter(args.tb_dir)

    # Load Data
    print("Loading Data")
    training_dataloader = load_data(
        data_dir=args.train_dir,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        smoketest=args.smoketest
    )

    val_dataloader = load_data(
        data_dir=args.val_dir,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        smoketest=args.smoketest
    )

    print(f"{len(training_dataloader)} batches per epoch")

    print("Initializing Model")
    model = construct_model(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    print("Beginning Model Training")
    train_model(model, training_dataloader, val_dataloader, optimizer, writer, args)


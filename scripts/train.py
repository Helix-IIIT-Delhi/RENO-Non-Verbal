#!/usr/bin/env python3
"""Training script for RENO"""

import argparse
import torch
import yaml
from reno import RENOModel


def train():
    parser = argparse.ArgumentParser(description='Train RENO model')
    parser.add_argument('--config', default='configs/default_config.yaml')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', default='./checkpoints')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model = RENOModel(**config['model'])
    print("Training RENO model...")
    print(f"Config: {config}")
    print(f"Data directory: {args.data_dir}")


if __name__ == '__main__':
    train()

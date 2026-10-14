#!/usr/bin/env python3
"""Inference script for RENO"""

import argparse
import torch
from reno import RENOModel


def inference():
    parser = argparse.ArgumentParser(description='Run RENO inference')
    parser.add_argument('--audio', required=True)
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()

    model = RENOModel(num_emotions=8)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    print(f"Audio: {args.audio}")
    print(f"Checkpoint: {args.checkpoint}")


if __name__ == '__main__':
    inference()

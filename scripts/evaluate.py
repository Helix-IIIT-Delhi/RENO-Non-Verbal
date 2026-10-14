#!/usr/bin/env python3
"""Evaluation script for RENO"""

import argparse
import torch
from reno import RENOModel


def evaluate():
    parser = argparse.ArgumentParser(description='Evaluate RENO model')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--test-data', required=True)
    args = parser.parse_args()

    model = RENOModel(num_emotions=8)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    print("Evaluating RENO model...")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test data: {args.test_data}")


if __name__ == '__main__':
    evaluate()

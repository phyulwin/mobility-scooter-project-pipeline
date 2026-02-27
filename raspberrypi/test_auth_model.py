"""
This test module validates the authentication model's processing pipeline on Raspberry Pi hardware.
It loads a pre-trained AuthModel and benchmarks its inference performance by running it repeatedly
on synthetic input data (shape: 1x27x128) while collecting timing and performance statistics.
The results help determine if the model meets real-time requirements for the mobility scooter authentication system.
"""

# Usage
'''
python raspberrypi/test_auth_model.py -r 100
'''

import torch
import argparse

import sys, os; sys.path.append(os.path.abspath('.'))
from pipeline.pipe.auth_model import AuthModel
from raspberrypi.utils import repeat_n_times_and_analysis


parser = argparse.ArgumentParser(prog='Authentication model Testing')
parser.add_argument('-r', '--repeat', type=int, required=True)
args = parser.parse_args()

mock_input = torch.randn(1, 27, 128)
process_func = AuthModel().process

@repeat_n_times_and_analysis(args.repeat)
def test_processing():
    process_func(mock_input)

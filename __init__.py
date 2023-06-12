'''
These imports are shared across all files.
Created by Basile Van Hoorick for TCOW.
'''

# Library imports.
import argparse
import collections
import collections.abc
import copy
import cv2
import imageio
import itertools
import joblib
import json
import lovely_numpy
import lovely_tensors
import matplotlib.colors
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import platform
import random
import rich
import rich.console
import rich.logging
import rich.progress
import scipy
import seaborn as sns
import shutil
import sklearn
import sklearn.decomposition
import sys
import time
import torch
import torch.nn
import torch.nn.functional
import torch.optim
import torch.utils
import torch.utils.data
import torchvision
import torchvision.datasets
import torchvision.io
import torchvision.models
import torchvision.transforms
import torchvision.utils
import tqdm
import tqdm.rich
import warnings
from collections import defaultdict
from einops import rearrange, repeat
from lovely_numpy import lo
from rich import print

PROJECT_NAME = 'tcow'

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'data/'))
sys.path.append(os.path.join(os.getcwd(), 'eval/'))
sys.path.append(os.path.join(os.getcwd(), 'model/'))
sys.path.append(os.path.join(os.getcwd(), 'third_party/'))
sys.path.append(os.path.join(os.getcwd(), 'utils/'))

lovely_tensors.monkey_patch()


# Quick functions for usage during debugging:

def mmm(x):
    return (x.min(), x.mean(), x.max())


def st(x):
    return (x.dtype, x.shape)


def stmmm(x):
    return (*st(x), *mmm(x))

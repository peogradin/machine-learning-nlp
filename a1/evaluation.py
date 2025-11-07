# %%
import torch, nltk, pickle
nltk.download('punkt_tab')
from torch import nn
from collections import Counter
from transformers import BatchEncoding, PretrainedConfig, PreTrainedModel

from torch.utils.data import DataLoader
import numpy as np
import sys, time, os
# %%
from A1_skeleton import *

# %%

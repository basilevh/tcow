'''
Neural network architecture description.
Created by Basile Van Hoorick for TCOW.
'''

import os
import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'seeker/'))

from __init__ import *

# Internal imports.
import mask_tracker


class Seeker(torch.nn.Module):

    def __init__(self, logger, **kwargs):
        super().__init__()
        self.logger = logger
        self.seeker = mask_tracker.QueryMaskTracker(logger, **kwargs)

    def forward(self, *args):
        return self.seeker(*args)

"""Launch the frozen revision experiment with bounded CPU threading."""
import numpy as np  # Load NumPy before Torch on this Windows installation.
import torch
from train_unified import main

if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    main()

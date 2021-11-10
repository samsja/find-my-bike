# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Test

# %load_ext autoreload
# %autoreload 2

# +
import os

os.chdir("..")
# -

from find_my_bike.executors import ResNetEncoder

ResNetEncoder(device="cpu", pretrain_path="data/models/v1/pytorch.ckpt")

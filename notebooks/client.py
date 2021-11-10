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

# %load_ext autoreload
# %autoreload 2

# +
import os

os.chdir("..")

from io import BytesIO

import matplotlib.pyplot as plt

# +
from jina import Client, Document, DocumentArray
from jina.types.document.generators import from_files
from jina.types.request import Response
from PIL import Image

from find_my_bike.executors import ResNetEncoder

# -


def print_result(resp: Response):
    print(resp.docs.get_attributes("matches"))
    resp.docs.plot_image_sprites()

    for doc in resp.docs:
        in_memory_file = BytesIO()
        doc.matches.plot_image_sprites(in_memory_file)
        plt.show(Image.open(in_memory_file))


query = DocumentArray(from_files("data/img.png"))

c = Client(protocol="http", port=12345)  # connect to localhost:12345

c.post(
    "/eval",
    query,
    shuffle=True,
    parameters={"top_k": 4},
    on_done=print_result,
    show_progress=True,
)

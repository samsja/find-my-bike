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

# + tags=[]
from itertools import chain

import numpy as np
from jina import Document, DocumentArray, Flow
from jina.types.document.generators import from_files

from find_my_bike.executors import KNNIndexer, ResNetEncoder

# -


# +
files = chain(
    chain(
        from_files("data/bike_data/bmx/*.png"),
        from_files("data/bike_data/vtt/*.png"),
    ),
    from_files("data/bike_data/course/*.png"),
)

docs = DocumentArray(files)[0:300]
query = DocumentArray(from_files("data/query/*.png"))

len(docs), len(query)
# -

query.plot_image_sprites()

f = (
    Flow()
    .add(name="encoder", uses=ResNetEncoder, uses_with={"device": "cuda"})
    .add(
        name="k_nn",
        uses=KNNIndexer,
    )
)
f.plot("data/flow.png")


def print_result(resp):
    print(resp.docs.get_attributes("matches"))
    resp.docs.plot_image_sprites()

    for doc in resp.docs:
        for match in doc.matches:
            doc.convert_uri_to_image_blob()
        doc.matches.plot_image_sprites()


# + tags=[]
with f:
    f.index(docs)
    f.post(
        "/eval",
        query,
        shuffle=True,
        parameters={"top_k": 40},
        on_done=print_result,
        show_progress=True,
    )

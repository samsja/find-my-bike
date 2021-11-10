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
# -

from jina import Client, DocumentArray
from jina.types.document.generators import from_files
from jina.types.request import Response


def print_result(resp: Response):
    print(resp.docs.get_attributes("matches"))
    resp.docs.plot_image_sprites()

    for doc in resp.docs:
        for match in doc.matches:
            doc.convert_uri_to_image_blob()
        doc.matches.plot_image_sprites()


query = DocumentArray(from_files("data/query/*.png"))

c = Client(protocol="http", port=12345)  # connect to localhost:12345

c.post(
    "/eval",
    query,
    shuffle=True,
    parameters={"top_k": 40},
    on_done=print_result,
    show_progress=True,
)

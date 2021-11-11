from io import BytesIO

import gradio as gr
import numpy as np
from jina import Client, Document, DocumentArray
from jina.types.document.converters import _move_channel_axis, _to_image_blob
from jina.types.document.generators import from_files
from jina.types.request import Response
from PIL import Image

c = Client(protocol="http", port=12345)
query = DocumentArray(from_files("data/query/*.png"))


class PrintResult:
    def __init__(self):
        self.__name__ = "PrintResult"
        self.imgs = []

    def __call__(self, resp: Response):

        doc = resp.docs[0]

        for match in doc.matches:
            match.convert_uri_to_image_blob()
            self.imgs.append(match.blob)


def create_query_from_img(img: np.array) -> DocumentArray:

    query = Document()
    query.blob = _move_channel_axis(img)
    return DocumentArray([query])


def search(img: np.array):

    query = create_query_from_img(img)
    print_result = PrintResult()
    c.post(
        "/eval",
        query,
        shuffle=True,
        parameters={"top_k": 4},
        on_done=print_result,
        show_progress=True,
    )

    return print_result.imgs


inputs = gr.inputs.Image()

outputs = gr.outputs.Carousel(["image"])

gr.Interface(fn=search, inputs=inputs, outputs=outputs).launch()

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from jina import DocumentArray, Executor, requests
from torchvision.models import resnet18


class ResNetEncoder(Executor):
    """
    Encode data using SVD decomposition
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Identity()
        self.model.eval()

    @requests
    def encode(self, docs: "DocumentArray", **kwargs):
        content = np.stack(docs.get_attributes("content"))
        embeds = np.zeros(content.shape[0])

        with torch.inference_mode():
            embeds = self.model(torch.from_numpy(content).float()).numpy()

        for doc, embed, cont in zip(docs, embeds, content):
            doc.embedding = embed
            doc.content = cont


class KNNIndexer(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @requests(on="/index")
    def index(self, docs: "DocumentArray", **kwargs):
        self.docs = docs

    @requests(on=["/search", "/eval"])
    def search(self, docs: "DocumentArray", parameters: Dict, **kwargs):
        docs.match(
            self.docs,
            metric="cosine",
            normalization=(1, 0),
            limit=int(parameters["top_k"]),
        )

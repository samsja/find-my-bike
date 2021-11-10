from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from jina import DocumentArray, Executor, requests
from vision_transformers.transformers.resnet import ResNet


class ResNetEncoder(Executor):
    """
    Encode data using SVD decomposition
    """

    def __init__(self, device: torch.device, pretrain_path: str, **kwargs):
        super().__init__(**kwargs)

        if pretrain_path is None:
            self.model = ResNet(num_classes=3, pretrained=True)
        else:
            self.model = ResNet(num_classes=3, pretrained=False)
            self.model.load_state_dict(torch.load(pretrain_path))

        self.model.fc = nn.Identity()
        self.model.eval()

        self.device = device
        self.model.to(self.device)

    @requests
    def encode(self, docs: "DocumentArray", **kwargs):

        for doc in docs:
            if doc.blob is None:
                doc.convert_uri_to_image_blob()
                original_blob = None
            else:
                original_blob = np.copy(doc.blob)

            doc.set_image_blob_shape(
                shape=(224, 224)
            ).set_image_blob_normalization().set_image_blob_channel_axis(-1, 0)

        content = np.stack(docs.get_attributes("content"))
        embeds = np.zeros(content.shape[0])

        with torch.inference_mode():
            embeds = (
                self.model(torch.from_numpy(content).float().to(self.device))
                .to("cpu")
                .numpy()
            )

        for doc, embed, cont in zip(docs, embeds, content):
            doc.embedding = embed
            doc.content = cont

            if original_blob is not None:
                doc.blob = original_blob
            else:
                doc.pop("blob")


class KNNIndexer(Executor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._docs = DocumentArray([])

    @requests(on="/index")
    def index(self, docs: "DocumentArray", **kwargs):
        self._docs.extend(docs)

    @requests(on=["/search", "/eval"])
    def search(self, docs: "DocumentArray", parameters: Dict, **kwargs):
        docs.match(
            self._docs,
            metric="cosine",
            limit=int(parameters["top_k"]),
        )

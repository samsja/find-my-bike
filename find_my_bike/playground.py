import numpy as np
from jina import Document, DocumentArray, Flow

from find_my_bike.executors import KNNIndexer, ResNetEncoder

query = DocumentArray([Document(content=np.zeros((3, 224, 224))) for _ in range(2)])

docs = DocumentArray([Document(content=np.zeros((3, 224, 224))) for _ in range(10)])

f = Flow().add(name="encoder", uses=ResNetEncoder).add(name="k_nn", uses=KNNIndexer)
f.plot("data/flow.png")

with f:
    f.index(docs)
    f.post(
        "/eval",
        query,
        shuffle=True,
        parameters={"top_k": 5},
        show_progress=True,
    )

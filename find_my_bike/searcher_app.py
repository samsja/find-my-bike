from itertools import chain

import numpy as np
from jina import Document, DocumentArray, Flow
from jina.types.document.generators import from_files

from find_my_bike.executors import KNNIndexer, ResNetEncoder


def searcher_app():

    files = chain(
        chain(
            from_files("data/bike_data/bmx/*.png"),
            from_files("data/bike_data/vtt/*.png"),
        ),
        from_files("data/bike_data/course/*.png"),
    )

    docs = DocumentArray(files)

    mask = np.arange(len(docs))
    np.random.shuffle(mask)

    docs = docs[mask.tolist()][0:500]

    query = DocumentArray(from_files("data/query/*.png"))

    f = (
        Flow(port_expose=12345, protocol="http", cors=True)
        .add(
            name="encoder",
            uses=ResNetEncoder,
            uses_with={
                "device": "cuda",
                "pretrain_path": "data/models/v1/pytorch.ckpt",
                "batch_size": 256,
            },
            replicas=2,
        )
        .add(
            name="k_nn",
            uses=KNNIndexer,
        )
    )

    with f:
        f.index(docs, request_size=300)

        f.block()


if __name__ == "__main__":
    searcher_app()

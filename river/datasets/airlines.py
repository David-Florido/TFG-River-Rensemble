import string
from river import stream

from . import base


class Airlines(base.FileDataset):

    def __init__(self):
        super().__init__(
            n_samples=1_250,
            n_features=9,
            task=base.BINARY_CLF,
            filename="airlines.arrf",
        )

    def __iter__(self):
        return stream.iter_arff(
            self.path,
            target="Delay")

"""     def __iter__(self):
        return stream.iter_arff(
            self.path,
            target="there_is_delay",
            converters={
                "airline": string,
                "flight": float,
                "airport_from": string,
                "airport_to": string,
                "anchor_from_other_domain": float,
                "day_of_week": int,
                "time": float,
                "length": float,
                "there_is_delay": lambda x: x == "1",
            },
        ) """
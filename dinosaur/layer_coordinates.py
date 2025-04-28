from __future__ import annotations
import dataclasses
import numpy as np


@dataclasses.dataclass(frozen=True)
class LayerCoordinates:
    layers: int

    @property
    def centers(self) -> np.ndarray:
        return np.arange(self.layers)

    def asdict(self):
        return dataclasses.asdict(self)

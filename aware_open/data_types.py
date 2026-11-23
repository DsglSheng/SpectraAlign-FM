from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass
class SampleMetadataBatch:
    modality: Sequence[str] | str
    sensor: Sequence[str] | str
    level: Sequence[str] | str
    gsd: torch.Tensor
    latitude: Optional[torch.Tensor] = None
    longitude: Optional[torch.Tensor] = None
    timestamp: Optional[torch.Tensor] = None
    footprint_area: Optional[torch.Tensor] = None


@dataclass
class ModalityBatch:
    values: torch.Tensor
    wavelengths: torch.Tensor
    fwhm: Optional[torch.Tensor]
    metadata: SampleMetadataBatch
    valid_mask: Optional[torch.Tensor] = None

    @property
    def batch_size(self) -> int:
        return int(self.values.shape[0])

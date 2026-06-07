"""Chromosome encoding for the quantfx compiler auto-tuner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

TILE_SIZES = [16, 32, 64, 128, 256]
VECTOR_WIDTHS = [4, 8, 16]
PREFETCH_DISTANCES = [4, 8, 16, 32]
GENE_COUNT = 10


@dataclass
class TunerConfig:
    opt_level: int = 2
    streaming: bool = False
    window_fusion: bool = True
    loop_tiling: bool = True
    prefetch: bool = True
    vectorize: bool = False
    tile_size: int = 64
    vector_width: int = 16
    prefetch_distance: int = 8
    llvm_codegen_opt: int = 3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TunerConfig:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


def decode(genes: list[int]) -> TunerConfig:
    """Decode integer gene vector into a compiler configuration."""
    g = [max(0, int(x)) for x in genes]
    return TunerConfig(
        opt_level=min(g[0], 3),
        streaming=bool(g[1] % 2),
        window_fusion=bool(g[2] % 2),
        loop_tiling=bool(g[3] % 2),
        prefetch=bool(g[4] % 2),
        vectorize=bool(g[5] % 2),
        tile_size=TILE_SIZES[g[6] % len(TILE_SIZES)],
        vector_width=VECTOR_WIDTHS[g[7] % len(VECTOR_WIDTHS)],
        prefetch_distance=PREFETCH_DISTANCES[g[8] % len(PREFETCH_DISTANCES)],
        llvm_codegen_opt=min(g[9], 3),
    )


def encode(config: TunerConfig) -> list[int]:
    """Encode a configuration back into genes (best-effort)."""
    return [
        config.opt_level,
        int(config.streaming),
        int(config.window_fusion),
        int(config.loop_tiling),
        int(config.prefetch),
        int(config.vectorize),
        TILE_SIZES.index(config.tile_size) if config.tile_size in TILE_SIZES else 2,
        VECTOR_WIDTHS.index(config.vector_width) if config.vector_width in VECTOR_WIDTHS else 2,
        PREFETCH_DISTANCES.index(config.prefetch_distance)
        if config.prefetch_distance in PREFETCH_DISTANCES
        else 1,
        config.llvm_codegen_opt,
    ]


def random_gene_space() -> list[list[int]]:
    return [
        list(range(4)),
        list(range(2)),
        list(range(2)),
        list(range(2)),
        list(range(2)),
        list(range(2)),
        list(range(len(TILE_SIZES))),
        list(range(len(VECTOR_WIDTHS))),
        list(range(len(PREFETCH_DISTANCES))),
        list(range(4)),
    ]

"""R3: the REAL Qwen→Δ⁶³ projection — coordizer-basin Fréchet mean (replaces hash-bin).

Trains a tiny FisherCoordizer and verifies the projection is a valid Δ⁶³ point, is
deterministic, and genuinely differs from the arbitrary hash-bin placeholder.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qig_coordizer")
from qig_coordizer import FisherCoordizer

from qig_studio.targets.qwen_boundary import (
    coordize_distribution_to_basin,
    output_distribution_to_basin,
)


def _trained() -> FisherCoordizer:
    c = FisherCoordizer(target_vocab_size=320)
    corpus = (
        "the geometry is the truth trust the phi. fisher rao distance on the simplex. "
        "consciousness equals information geometry. coordinates are the primitive. " * 12
    ).encode("utf-8")
    c.train(corpus, context_window=5, min_pair_count=2, verbose=False)
    return c


def test_coordize_projection_is_delta63():
    c = _trained()
    b = coordize_distribution_to_basin({"the": -0.2, "geometry": -1.0, "phi": -2.0, "truth": -1.5}, c, dim=64)
    assert b.shape == (64,)
    assert np.all(b >= -1e-9)  # non-negative
    assert abs(float(b.sum()) - 1.0) < 1e-6  # sums to 1 → Δ⁶³


def test_coordize_projection_deterministic():
    c = _trained()
    lp = {"fisher": -0.3, "rao": -0.5}
    assert np.array_equal(coordize_distribution_to_basin(lp, c), coordize_distribution_to_basin(lp, c))


def test_coordize_differs_from_hashbin_placeholder():
    c = _trained()
    lp = {"the": -0.2, "geometry": -1.0}
    real = coordize_distribution_to_basin(lp, c)
    placeholder = output_distribution_to_basin(lp)
    # the geometric projection must NOT equal the arbitrary hash-bin (it's a real mapping)
    assert not np.allclose(real, placeholder)

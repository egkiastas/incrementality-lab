from __future__ import annotations

import pytest

from incrementality.synthetic import SyntheticExperiment, generate


@pytest.fixture(scope="session")
def big_experiment() -> SyntheticExperiment:
    return generate(200_000, seed=11)


@pytest.fixture(scope="session")
def holdout_experiment() -> SyntheticExperiment:
    return generate(40_000, seed=12)

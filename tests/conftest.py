import pytest


def seed_all(seed: int) -> None:
    """Set the seed for reproducibility."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@pytest.fixture(scope="session")
def seed() -> int:
    """Fixture to provide a fixed seed for reproducibility."""
    return 0

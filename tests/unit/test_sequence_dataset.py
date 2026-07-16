import numpy as np
import pytest

from aeroguard.deep.sequence_dataset import SequenceWindowDataset


def test_sequence_window_dataset_derives_lengths_from_mask() -> None:
    sequences = np.zeros((2, 4, 3), dtype=np.float32)
    sequences[0, -2:, -1] = 1.0
    sequences[1, :, -1] = 1.0

    dataset = SequenceWindowDataset(sequences, np.array([5.0, 7.0], dtype=np.float32))
    x, y, length = dataset[0]

    assert len(dataset) == 2
    assert tuple(x.shape) == (4, 3)
    assert float(y.item()) == 5.0
    assert int(length.item()) == 2
    assert int(dataset[1][2].item()) == 4


def test_sequence_window_dataset_validates_alignment_and_non_empty() -> None:
    with pytest.raises(ValueError, match="aligned"):
        SequenceWindowDataset(np.zeros((2, 4, 3), dtype=np.float32), np.array([1.0], dtype=np.float32))
    with pytest.raises(ValueError, match="must not be empty"):
        SequenceWindowDataset(np.zeros((0, 4, 3), dtype=np.float32), np.array([], dtype=np.float32))


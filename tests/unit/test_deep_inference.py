import numpy as np
import torch

from aeroguard.deep.inference import predict_batches
from aeroguard.deep.sequence_dataset import SequenceWindowDataset


class _SignedModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return x[:, -1:, 0] - 1.0


def test_predict_batches_preserves_order_and_clips_negative_predictions() -> None:
    sequences = np.zeros((3, 2, 3), dtype=np.float32)
    sequences[:, :, -1] = 1.0
    sequences[:, -1, 0] = np.array([0.5, 2.0, 4.0], dtype=np.float32)
    dataset = SequenceWindowDataset(sequences, np.array([1.0, 1.0, 1.0], dtype=np.float32))

    predictions = predict_batches(_SignedModel(), dataset, torch.device("cpu"), batch_size=2)

    np.testing.assert_allclose(predictions, [0.0, 1.0, 3.0])


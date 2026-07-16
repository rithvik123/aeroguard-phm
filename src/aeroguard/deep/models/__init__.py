"""Deep RUL model registry."""

from aeroguard.deep.models.cnn1d import CNN1DRegressor
from aeroguard.deep.models.cnn_lstm import CNNLSTMRegressor
from aeroguard.deep.models.gru import GRURegressor
from aeroguard.deep.models.lstm import LSTMRegressor
from aeroguard.deep.models.patch_transformer import PatchTemporalTransformerRegressor
from aeroguard.deep.models.physics_guided_patch_transformer import PhysicsGuidedPatchTransformer
from aeroguard.deep.models.sequence_mlp import SequenceMLPRegressor
from aeroguard.deep.models.tcn import TCNRegressor
from aeroguard.deep.models.temporal_transformer import TemporalTransformerRegressor

MODEL_CLASSES = {
    "sequence_mlp": SequenceMLPRegressor,
    "cnn1d": CNN1DRegressor,
    "lstm": LSTMRegressor,
    "gru": GRURegressor,
    "tcn": TCNRegressor,
    "cnn_lstm": CNNLSTMRegressor,
    "temporal_transformer": TemporalTransformerRegressor,
    "patch_transformer": PatchTemporalTransformerRegressor,
    "physics_guided_patch_transformer": PhysicsGuidedPatchTransformer,
}

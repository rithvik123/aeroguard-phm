"""Physics-guided degradation constraints for Phase 5C RUL modelling."""

from aeroguard.deep.physics.candidate_registry import default_candidate_registry, validate_candidate_registry
from aeroguard.deep.physics.composite_loss import CompositePhysicsLoss, PhysicsLossConfig
from aeroguard.deep.physics.paired_sequences import TemporalPairingConfig, build_temporal_pairs
from aeroguard.deep.physics.regime_consistency import RegimePairingConfig, build_regime_pairs

__all__ = [
    "CompositePhysicsLoss",
    "PhysicsLossConfig",
    "TemporalPairingConfig",
    "RegimePairingConfig",
    "build_temporal_pairs",
    "build_regime_pairs",
    "default_candidate_registry",
    "validate_candidate_registry",
]

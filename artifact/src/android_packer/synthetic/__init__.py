"""Synthetic APK generation utilities."""

from android_packer.synthetic.packer import (
    InjectedPayload,
    SUPPORTED_TRANSFORMS,
    SyntheticBuildResult,
    SyntheticPackerError,
    build_synthetic_apk,
)
from android_packer.synthetic.seed_derivation import (
    KNUTH_MULTIPLIER,
    derive_task_rng_seed,
)
from android_packer.synthetic.transforms import (
    TRANSFORMS,
    TransformContext,
    TransformFn,
    register_transform,
)

__all__ = [
    "InjectedPayload",
    "KNUTH_MULTIPLIER",
    "SUPPORTED_TRANSFORMS",
    "SyntheticBuildResult",
    "SyntheticPackerError",
    "TRANSFORMS",
    "TransformContext",
    "TransformFn",
    "build_synthetic_apk",
    "derive_task_rng_seed",
    "register_transform",
]

# Copyright (c) 2021-2025, InterDigital Communications, Inc
# All rights reserved.
# BSD 3-Clause Clear License (see LICENSE file)

"""Configuration loading and validation for CompressAI CLI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from argparse import Namespace
from typing import Any

import tomli


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    type: str = "Adam"
    lr: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OptimizerConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""

    type: str = "ReduceLROnPlateau"
    mode: str = "min"
    factor: float = 0.5
    patience: int = 20
    min_lr: float = 1e-7

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SchedulerConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelConfig:
    """Model configuration."""

    name: str = "mbt2018-mean"
    quality: int = 3
    pretrained: bool = False
    kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        kwargs = {k: v for k, v in d.items() if k not in cls.__dataclass_fields__}
        base: dict[str, Any] = {
            k: v for k, v in d.items() if k in cls.__dataclass_fields__
        }
        base["kwargs"] = {**base.get("kwargs", {}), **kwargs}
        return cls(**base)


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    path: str = ""
    split_train: str = "train"
    split_test: str = "test"
    patch_size: list[int] = field(default_factory=lambda: [256, 256])
    num_workers: int = 4

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DatasetConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingConfig:
    """Training configuration."""

    epochs: int = 100
    batch_size: int = 16
    test_batch_size: int = 64
    lmbda: float = 0.01
    metric: str = "mse"  # mse or ms-ssim
    clip_max_norm: float = 1.0
    seed: int | None = None
    cuda: bool = True
    save: bool = True
    save_dir: str = "./checkpoints"
    checkpoint: str | None = None
    log_interval: int = 10

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Config:
    """Complete training configuration."""

    domain: str = "image"  # image, video, pointcloud
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimizer_net: OptimizerConfig = field(default_factory=OptimizerConfig)
    optimizer_aux: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(lr=1e-3)
    )
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        """Create config from dictionary."""
        config = cls()

        if "domain" in d:
            config.domain = d["domain"]

        if "model" in d:
            config.model = ModelConfig.from_dict(d["model"])

        if "dataset" in d:
            config.dataset = DatasetConfig.from_dict(d["dataset"])

        if "training" in d:
            config.training = TrainingConfig.from_dict(d["training"])

        if "optimizer" in d:
            opt = d["optimizer"]
            if "net" in opt:
                config.optimizer_net = OptimizerConfig.from_dict(opt["net"])
            if "aux" in opt:
                config.optimizer_aux = OptimizerConfig.from_dict(opt["aux"])

        if "scheduler" in d:
            config.scheduler = SchedulerConfig.from_dict(d["scheduler"])

        return config

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        """Load configuration from YAML, JSON, or TOML file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        suffix = path.suffix.lower()
        content = path.read_text()

        if suffix in (".yaml", ".yml"):
            try:
                import yaml

                data = yaml.safe_load(content)
            except ImportError:
                raise ImportError(
                    "PyYAML is required for YAML config files: pip install pyyaml"
                )
        elif suffix == ".json":
            data = json.loads(content)
        elif suffix == ".toml":
            data = tomli.loads(content)
        else:
            raise ValueError(f"Unsupported config format: {suffix}")

        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "domain": self.domain,
            "model": {
                "name": self.model.name,
                "quality": self.model.quality,
                "pretrained": self.model.pretrained,
                **self.model.kwargs,
            },
            "dataset": {
                "path": self.dataset.path,
                "split_train": self.dataset.split_train,
                "split_test": self.dataset.split_test,
                "patch_size": self.dataset.patch_size,
                "num_workers": self.dataset.num_workers,
            },
            "training": {
                "epochs": self.training.epochs,
                "batch_size": self.training.batch_size,
                "test_batch_size": self.training.test_batch_size,
                "lmbda": self.training.lmbda,
                "metric": self.training.metric,
                "clip_max_norm": self.training.clip_max_norm,
                "seed": self.training.seed,
                "cuda": self.training.cuda,
                "save": self.training.save,
                "save_dir": self.training.save_dir,
                "checkpoint": self.training.checkpoint,
            },
            "optimizer": {
                "net": {
                    "type": self.optimizer_net.type,
                    "lr": self.optimizer_net.lr,
                },
                "aux": {
                    "type": self.optimizer_aux.type,
                    "lr": self.optimizer_aux.lr,
                },
            },
            "scheduler": {
                "type": self.scheduler.type,
                "mode": self.scheduler.mode,
                "factor": self.scheduler.factor,
                "patience": self.scheduler.patience,
            },
        }

    def merge_cli_args(self, args: Namespace) -> "Config":
        """Merge CLI arguments into config (CLI takes precedence)."""
        if hasattr(args, "model") and args.model:
            self.model.name = args.model
        if hasattr(args, "quality") and args.quality is not None:
            self.model.quality = args.quality
        if hasattr(args, "dataset") and args.dataset:
            self.dataset.path = args.dataset
        if hasattr(args, "patch_size") and args.patch_size is not None:
            self.dataset.patch_size = args.patch_size
        if hasattr(args, "epochs") and args.epochs is not None:
            self.training.epochs = args.epochs
        if hasattr(args, "batch_size") and args.batch_size is not None:
            self.training.batch_size = args.batch_size
        if hasattr(args, "lmbda") and args.lmbda is not None:
            self.training.lmbda = args.lmbda
        if hasattr(args, "learning_rate") and args.learning_rate is not None:
            self.optimizer_net.lr = args.learning_rate
        if hasattr(args, "cuda") and args.cuda is not None:
            self.training.cuda = args.cuda
        if hasattr(args, "checkpoint") and args.checkpoint:
            self.training.checkpoint = args.checkpoint
        if hasattr(args, "seed") and args.seed is not None:
            self.training.seed = args.seed
        if hasattr(args, "save_dir") and args.save_dir:
            self.training.save_dir = args.save_dir
        return self

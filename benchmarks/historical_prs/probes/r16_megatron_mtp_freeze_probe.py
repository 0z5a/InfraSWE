#!/usr/bin/env python3
"""Check R16 MTP-only parameter and router-bias ownership without TE."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from megatron.training.training import (
    _add_model_freeze_pre_wrap_hook,
    _freeze_base_model_for_mtp,
)


class Router(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.expert_bias = torch.zeros(2)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = Router()
        self.mtp = torch.nn.Module()
        self.mtp.layers = torch.nn.ModuleList([Router()])


def main() -> None:
    model = TinyModel()
    _freeze_base_model_for_mtp([model])
    ownership = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    assert ownership == {
        "backbone.weight": False,
        "mtp.layers.0.weight": True,
    }, ownership
    assert model.backbone.frozen_expert_bias is True
    assert model.mtp.layers[0].frozen_expert_bias is False

    config = SimpleNamespace(pre_wrap_hooks=[])
    args = SimpleNamespace(freeze_all_layers=False, freeze_base_model_for_mtp=True)
    _add_model_freeze_pre_wrap_hook(config, args)
    _add_model_freeze_pre_wrap_hook(config, args)
    assert config.pre_wrap_hooks == [_freeze_base_model_for_mtp]
    print("mtp_only_parameter_and_router_ownership=pass")


if __name__ == "__main__":
    main()

# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import logging
import os
from copy import deepcopy
from functools import cached_property
from typing import Any
from typing import Literal

import numpy as np

from anemoi.inference.lazy import torch

from ..checkpoint import Checkpoint, MultiDomainCheckpoint
from ..decorators import main_argument
from ..runner import Runner
from ..runners.default import DefaultRunner
from ..runners.external_graph import ExternalGraphRunner
from . import runner_registry
from .external_graph import (
    update_state_dict,
    get_updated_supporting_arrays
)

LOG = logging.getLogger(__name__)

class MultiDomainMixin:
    domain: str
    
    @cached_property
    def checkpoint(self) -> Checkpoint:
        return MultiDomainCheckpoint(
            self.domain,
            self.config.checkpoint,
            patch_metadata=self.config.patch_metadata
        ) 
    def predict_step(self, model: "torch.nn.Module", input_tensor_torch: "torch.Tensor", **kwargs: Any) -> "torch.Tensor":
        return model.predict_step(input_tensor_torch, graph_label=self.domain, **kwargs)

class External(MultiDomainMixin,ExternalGraphRunner):
    def __init__(
        self,
        config: dict,
        domain: str,
        graph: str,
        *,
        output_mask: dict | None = {},
        graph_dataset: Any | None = None,
        update_supporting_arrays: dict[Literal["graph", "file"], dict[str, str]] | None = None,
        updated_number_of_grid_points: str | int | None = None,
        check_state_dict: bool | None = True,
        **kwargs: Any
    ) -> None:
        self.domain = domain

        super().__init__(
            config, 
            graph, 
            output_mask=output_mask, 
            graph_dataset=graph_dataset, 
            update_supporting_arrays=update_supporting_arrays, 
            updated_number_of_grid_points=updated_number_of_grid_points, 
            check_state_dict=check_state_dict
            )
        
    @cached_property
    def model(self) -> "torch.nn.Module":
        # load the model from the checkpoint
        device = self.device
        self.device = "cpu"
        model_instance = super().model
        state_dict_ckpt = deepcopy(model_instance.state_dict())

        # rebuild the model with the new graph
        model_instance.graph_data[self.domain] = self.graph
        model_instance.config = self.checkpoint._metadata._config
        model_instance._build_model()

        # reinstate the weights, biases and normalizer from the checkpoint
        # reinstating the normalizer is necessary for checkpoints that were created
        # using transfer learning, where the statistics as stored in the checkpoint
        # do not match the statistics used to build the normalizer in the checkpoint.
        model_instance = update_state_dict(
            model_instance, state_dict_ckpt, keywords=["bias", "weight", "processors.normalizer"]
        )

        LOG.info(f"Successfully built model with external graph and reassigned model weights with domain: {self.domain}!")
        self.device = device
        return model_instance.to(self.device)
    
    # def predict_step(self, input_tensor_torch: "torch.Tensor", **kwargs: Any) -> "torch.Tensor":
    #     return super().model.predict_step(input_tensor_torch, graph_label = self.domain, **kwargs)


class Internal(MultiDomainMixin,DefaultRunner):
    def __init__(self, config: dict, domain: str, **kwargs: Any):
        self.domain = domain

        super().__init__(config)


    @cached_property
    def model(self) -> "torch.nn.Module":
        # validate lazily
        _model = super().model

        if self.domain not in _model.graph_data:
            raise KeyError(
                f"Domain: {self.domain} not found in model.graph_data."
                f"Available internal domains: {list(_model.graph_data.keys())}"
            )
        return _model
        
@runner_registry.register("multi_domain")
@main_argument("domain")
class MultiDomain(DefaultRunner):
    def __new__(cls, config: dict, *args: list, **kwargs: dict) -> None:
        #if "graph" in kwargs:
        #    LOG.info("Using external graph with Multi Domain")
        #    return External(config, *args, **kwargs)
        #LOG.info("External graph is not provided for multi-domain, using iternal")
        # print(type(Internal(config, *args, **kwargs)))
        # exit()
        _use_external = "graph" in kwargs
        multi_domain_instance = External if _use_external else Internal

        Chosen = type(
            f"{cls.__name__}{'External' if _use_external else 'Internal'}",
            (cls, multi_domain_instance),
            {}
        )
        LOG.info(f"Multi Domain is using: {cls.__name__}{'External' if _use_external else 'Internal'}")
        return object.__new__(Chosen) #Internal(config, *args, **kwargs)

    

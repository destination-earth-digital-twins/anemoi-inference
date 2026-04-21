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
from collections import defaultdict
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

def check_all_domains(metadata: dict[str, Any], key: str) -> bool:
    """Check if a specific key is present in the metadata for all domains."""
    current_domain_value = metadata[next(iter(metadata))][key]
    assert all(
        current_domain_value == domain_metadata[key]
        for _, domain_metadata in metadata.items()
    ), f"Value for key '{key}' is not the same across all domains in the metadata."


def retrieve_domain_metadata(metadata: dict[str, Any], key: str) -> Any:
    """Retrieve metadata for a specific domain from a multi-domain metadata dictionary."""

    check_all_domains(metadata, key)
    first_domain = next(iter(metadata))
    return metadata[first_domain][key]

def get_pl(variable_name: str) ->tuple[str, int] | tuple[str,None]:
    assert isinstance(variable_name,str), f"expected type str, but got type {type(variable_name)}"
    split = variable_name.split("_")
    if len(split) > 1 and split[-1].isdigit():
        return split[0], int(split[-1])
    return split[0], None

def construct_variable_metadata(variables: list[str]) -> dict:
    """
    TODO: implement a more robust and general for decoding variable names:
    deciding pl level -> easy/done
    decide sfc vs constant a bit tricky...

    Function to recreate a simplified version of MARS keys
    for an external graph setup.

    variables: list[str] 
        Contains the variables used during training. 
        These are being used to determine MARS keys
    
    return:
        a dict containing metadata with MARS keys
    """
    assert len(variables) > 0 #, "Something went wrong"

    variable_metadata = defaultdict(dict)
    for var in variables:
        if "_" in var:
            param, levelist = get_pl(var)
            if levelist:
                variable_metadata[var]["mars"] = {
                    "param" : param,
                    "levtype": "pl",
                    "levelist": levelist,
                }
            else:
                # decide sfc or constant 
                print(var.split("_"))
                # param, levelist = var.split("_")
                # if levelist.isdigit():
                #     levtype = "pl"

                #     variable_metadata[var]["mars"] = {
                #         "param" : param,
                #         "levtype": levtype,
                #         "levelist": int(levelist),
                #     }
                # # else:
                variable_metadata[var]["mars"] = {
                    "param" : var,
                    "levtype": "sfc",
                }
    return variable_metadata


class External(MultiDomainMixin,DefaultRunner): #ExternalGraphRunner):
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
            # graph, 
            # output_mask=output_mask, 
            # graph_dataset=graph_dataset, 
            # update_supporting_arrays=update_supporting_arrays, 
            # updated_number_of_grid_points=updated_number_of_grid_points, 
            # check_state_dict=check_state_dict
            )
        #print(self.checkpoint._metadata._dataset["ARA"].variables_metadata==self.checkpoint._metadata._dataset["ARA"].variables_metadata)
        self.check_state_dict = check_state_dict
        self.graph_path = graph

        shape = self.graph["data"].x.shape[0]
        
        _variables = retrieve_domain_metadata(
                self.checkpoint._metadata._dataset, "variables"
            )
        _variables_metadata = construct_variable_metadata(_variables)
        
        self.checkpoint._metadata._dataset[self.domain] = {
            "variables": _variables,
            "variables_metadata": self.checkpoint._metadata._dataset["ARA"].variables_metadata,
            "frequency": retrieve_domain_metadata(
                self.checkpoint._metadata._dataset, "frequency"
            ),
            "supporting_arrays": get_updated_supporting_arrays(
                update_supporting_arrays, self.graph
            ),
            "dtype": retrieve_domain_metadata(
                self.checkpoint._metadata._dataset, "dtype"
            ),
            "shape": shape,
        }

        self.checkpoint._supporting_arrays[self.domain] = get_updated_supporting_arrays(
            update_supporting_arrays, self.graph
        )

        if output_mask:
            nodes = output_mask["nodes_name"]
            attribute = output_mask["attribute_name"]
            print(self.checkpoint._supporting_arrays[self.domain],self.checkpoint._supporting_arrays.keys())
            exit()
            self.checkpoint._supporting_arrays[self.domain]["output_mask"] = (
                self.graph[nodes][attribute].numpy().squeeze()
            )
            LOG.info(
                "Moving attribute '%s' of nodes '%s' from external graph to supporting arrays as 'output_mask'.",
                attribute,
                nodes,
            )

        if updated_number_of_grid_points is not None:
            if isinstance(updated_number_of_grid_points, str):
                updated_number_of_grid_points = len(
                    self.graph["data"][updated_number_of_grid_points]
                )
            self.checkpoint._metadata.number_of_grid_points = (
                updated_number_of_grid_points
            )
            LOG.info(
                "Updated number of grid points in the checkpoint metadata to %s.",
                updated_number_of_grid_points,
            )
    @cached_property
    def graph(self):

        graph_path = self.graph_path
        assert os.path.isfile(
            graph_path
        ), f"No graph found at {graph_path}. An external graph needs to be specified in the config file for this runner."
        LOG.info("Loading external graph from path %s.", graph_path)
        return torch.load(graph_path, map_location="cpu", weights_only=False)

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

    

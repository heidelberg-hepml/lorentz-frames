from typing import Dict, List, Union

import torch
from torch_geometric.typing import PairTensor

from tensorframes.lframes import LFrames
from tensorframes.nn.embedding.radial import compute_edge_vec
from tensorframes.nn.mlp import MLPWrapped
from tensorframes.nn.tfmessage_passing import TFMessagePassing
from tensorframes.reps import Irreps, TensorReps
from experiments.logger import LOGGER


class EdgeConv(TFMessagePassing):
    """Multi-Layer Perceptron Convolutional layer for graph neural networks.

    Attributes:
        in_reps (Union[TensorReps, Irreps, str]): Input tensor representations or irreps.
        radial_module (torch.nn.Module): Radial module.
        angular_module (torch.nn.Module): Angular module.
        concatenate_edge_vec (bool): Whether to concatenate edge vectors.
        concatenate_receiver_features_in_mlp1 (bool): Whether to concatenate receiver features in MLP1.
        concatenate_receiver_features_in_mlp2 (bool): Whether to concatenate receiver features in MLP2.
        mlp1 (MLP): First MLP layer.
        mlp2 (MLP): Second MLP layer.
        out_dim (int): Number of output channels.
        edge_feature_product_layer (torch.nn.Linear): Edge feature product layer.
    """

    def __init__(
        self,
        in_reps: Union[TensorReps, Irreps],
        hidden_channels: List[int],
        out_channels: int,
        aggr: str = "add",
        radial_module: torch.nn.Module = None,
        angular_module: torch.nn.Module = None,
        concatenate_receiver_features_in_mlp1: bool = True,
        **mlp_kwargs: Dict
    ):
        """
        Initialize the MLPConv layer.
        This module can be used to implement the following layers:
        MLP2(f_i, aggr(MLP1(f_i, transformed f_j, radial_embedding, angular_embedding)))
        or alternatively:
        MLP2(f_i, aggr(MLP1(f_i, transformed f_j) odot linear(radial_embedding, angular_embedding))).

        Args:
            in_reps (Union[TensorReps, Irreps]): Input tensor representations or irreps.
            hidden_channels (list[int]): List of hidden channel sizes.
            out_channels (int): Number of output channels.
            aggr (str, optional): Aggregation method. Defaults to "add".
            spatial_dim (int, optional): Spatial dimension. Defaults to 3.
            second_hidden_channels (List[int], optional): List of hidden channel sizes for the second MLP. Defaults to None.
            radial_module (torch.nn.Module, optional): Radial module. Defaults to None.
            angular_module (torch.nn.Module, optional): Angular module. Defaults to None.
            concatenate_edge_vec (bool, optional): Whether to concatenate edge vectors. Defaults to False.
            concatenate_receiver_features_in_mlp1 (bool, optional): Whether to concatenate receiver features in MLP1. Defaults to False.
            concatenate_receiver_features_in_mlp2 (bool, optional): Whether to concatenate receiver features in MLP2. Defaults to False.
            use_edge_feature_product (bool, optional): Whether to use edge feature product. Defaults to False.
            **mlp_kwargs: Additional keyword arguments for the MLP layers.
        """
        self.in_reps = in_reps
        super().__init__(aggr=aggr, params_dict={"x": {"type": "local", "rep": self.in_reps}}) #check this out

        self.radial_module = radial_module
        self.angular_module = angular_module
        self.concatenate_receiver_features_in_mlp1 = concatenate_receiver_features_in_mlp1

        mlp1_in_dim = self.in_reps.dim+radial_module.out_dim+angular_module.out_dim

        if concatenate_receiver_features_in_mlp1:
            mlp1_in_dim += self.in_reps.dim

        LOGGER.info(f"{mlp1_in_dim=}")

        self.mlp1 = MLPWrapped(
        in_channels=mlp1_in_dim,
        hidden_channels=hidden_channels + [out_channels],
        **mlp_kwargs
        )

        self.out_dim = out_channels
        
    def forward(
        self,
        x: Union[torch.Tensor, PairTensor],
        pos: Union[torch.Tensor, PairTensor],
        lframes: Union[LFrames, PairTensor],
        batch: Union[torch.Tensor, PairTensor],
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the MLPConv layer.

        Args:
            x (Union[torch.Tensor, PairTensor]): Input node features.
            pos (Union[torch.Tensor, PairTensor]): Node positions.
            lframes (Union[LFrames, PairTensor]): Local frames.
            batch (Union[torch.Tensor, PairTensor]): Batch indices.
            edge_index (torch.Tensor): Edge indices.

        Returns:
            torch.Tensor: Output node features.
        """
        if not isinstance(x, tuple):
            x = (x, x)
        if not isinstance(pos, tuple):
            pos = (pos, pos)
        if not isinstance(lframes, tuple):
            lframes = (lframes, lframes)
        if not isinstance(batch, tuple):
            batch = (batch, batch)

        edge_vec = compute_edge_vec(pos=pos, edge_index=edge_index, lframes=lframes)
        
        if self.radial_module is None:
            radial_embedding = None
        else:
            radial_embedding = self.radial_module(edge_vec=edge_vec)
        if self.angular_module is None:
            angular_embedding = None
        else:
            angular_embedding = self.angular_module(edge_vec=edge_vec)

        batch = (batch[0].view(-1, 1), batch[1].view(-1, 1))  # needed for index-magic
        #LOGGER.info(f"preprop {x[0].shape=}")
        x_aggr = self.propagate(
            edge_index,
            x=x,
            batch=batch,
            lframes=lframes,
            edge_vec=edge_vec,
            radial_embedding=radial_embedding,
            angular_embedding=angular_embedding,
        )

        return x_aggr

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        batch_i: torch.Tensor,
        batch_j: torch.Tensor,
        edge_vec: torch.Tensor,
        radial_embedding: torch.Tensor,
        angular_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Message passing function of the MLPConv layer.

        Args:
            x_i (torch.Tensor): Input node features of the receiver.
            x_j (torch.Tensor): Input node features of the sender.
            batch_i (torch.Tensor): Batch indices of the receiver.
            batch_j (torch.Tensor): Batch indices of the sender.
            edge_vec (torch.Tensor): Edge vectors.
            radial_embedding (torch.Tensor): Radial embeddings.
            angular_embedding (torch.Tensor): Angular embeddings.

        Returns:
            torch.Tensor: Output node features.
        """
        assert torch.allclose(batch_i, batch_j), "batch_i and batch_j must be equal"

        #LOGGER.info(f"{x_i.shape=}")
        if self.concatenate_receiver_features_in_mlp1:
            x = torch.cat((x_i, x_j-x_i), dim=-1) #standart edgeConv
        else:
            x = x_j #this is pretty dumb
        #LOGGER.info(f"{x.shape=}")
        edge_features = None

        if radial_embedding is not None:
            edge_features = (
                radial_embedding
            )
        #LOGGER.info(f"{radial_embedding.shape=}")
        #LOGGER.info(f"{angular_embedding.shape=}")

        if angular_embedding is not None:
            edge_features = (
                angular_embedding
                if edge_features is None
                else torch.cat((edge_features, angular_embedding), dim=-1)
            )

        if edge_features is None:
            assert x is not None, "x and edge_features are both None"
        else:
            if x is None:
                x = edge_features
            else:
                x = torch.cat((x, edge_features), dim=-1)
        #LOGGER.info(f"before mlp: {x.shape=}, mlp: {self.mlp1=}")
        x = self.mlp1(x, batch=batch_i.view(-1))
        #LOGGER.info(f"after mlp: {x.shape=}")
        return x

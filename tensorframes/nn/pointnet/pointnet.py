from typing import Tuple, Union

import torch

from tensorframes.lframes import LFrames
from tensorframes.lframes.learning_lframes import WrappedLearnedLFrames
from tensorframes.lframes.updating_lframes import QuaternionsUpdateLFrames
from tensorframes.nn.edge_conv import EdgeConv
from tensorframes.nn.embedding.axial import AxisWiseEmbeddingFromRadial
from tensorframes.nn.embedding.radial import GaussianEmbedding
from tensorframes.nn.local_global import FromGlobalToLocalFrame, FromLocalToGlobalFrame
from tensorframes.nn.mlp import MLPWrapped
from tensorframes.nn.pointnet.dgcnn_blocks import DynamicSAModule
from tensorframes.nn.pointnet.pointnet_blocks import (
    FinalLframesLayer,
    FPModule,
    GlobalSAModule,
    SAModule,
)
from tensorframes.reps import Irreps, TensorReps
from tensorframes.reps.utils import parse_reps
from tensorframes.utils import consistent_length_check, repeat_in_list
from tensorframes.utils.point_sampling import CustomPointSampler


class PointNetEncoder(torch.nn.Module):
    """Pointnet++ Encoder module."""

    def __init__(
        self,
        list_in_reps: list[Union[TensorReps, Irreps, str]],
        list_hidden_channels: list[list[int]],
        list_center_sampler: list[CustomPointSampler],
        sam_out_reps: Union[TensorReps, Irreps, str],
        list_second_hidden_channels: list[list[int]] | None = None,
        list_dynamic_k: list[int] | None = None,
        list_r: list[float] | None = None,
        list_conv_kwargs: Union[dict, list[dict]] | None = None,
        list_sam_kwargs: Union[dict, list[dict]] | None = None,
        list_lframes_update_kwargs: Union[dict, list[dict]] | None = None,
        first_concats_pos_to_features: bool = False,
        first_concatenate_edge_vec: bool = False,
        use_dynamic_sam: bool = False,
        lframes_learner: WrappedLearnedLFrames | None = None,
        global_sam: GlobalSAModule | None = None,
        cache_layer_outputs: bool = False,
        adjust_max_radius_with_sampling: bool = True,
        radial_module_type: str = "none",
        radial_module_kwargs: dict = None,
        shared_radial_module: bool = False,
        convert_radial_to_axial: bool = False,
        axial_from_radial_kwargs: dict = None,
        spatial_dim: int = 3,
    ) -> None:
        """Initializes the Pointnet++ Encoder module.

        Args:
            list_in_reps (list[Union[TensorReps, Irreps, str]]): List of input representations for each layer.
            list_hidden_channels (list[list[int]]): List of hidden channel sizes for each layer.
            list_center_sampler (list[CustomPointSampler]): List of custom point samplers for each layer.
            sam_out_reps (Union[TensorReps, Irreps, str]): Output representations of the point sampler.
            list_second_hidden_channels (list[list[int]] | None, optional): List of second hidden channel sizes for each layer. Defaults to None.
            list_dynamic_k (list[int] | None, optional): List of dynamic k values for each layer. Defaults to None.
            list_r (list[float] | None, optional): List of r values for each layer. Defaults to None.
            list_conv_kwargs (Union[dict, list[dict]] | None, optional): Convolutional layer keyword arguments. Defaults to None.
            list_sam_kwargs (Union[dict, list[dict]] | None, optional): Point sampler keyword arguments. Defaults to None.
            list_lframes_update_kwargs (Union[dict, list[dict]] | None, optional): Local frames update keyword arguments. Defaults to None.
            first_concats_pos_to_features (bool, optional): Whether the first layer concatenates position to features. Defaults to False.
            first_concatenate_edge_vec (bool, optional): Whether the first layer concatenates edge vectors. Defaults to False.
            use_dynamic_sam (bool, optional): Whether to use dynamic sampling. Defaults to False.
            lframes_learner (WrappedLearnedLocalFramesModule | None, optional): Learned local frames module. Defaults to None.
            global_sam (GlobalSAModule | None, optional): Global point sampler module. Defaults to None.
            cache_layer_outputs (bool, optional): Whether to cache layer outputs. Defaults to False.
            adjust_max_radius_with_sampling (bool, optional): Whether to adjust the maximum radius with sampling. Defaults to True.
            radial_module_type (str, optional): Type of radial module. Defaults to "none".
            radial_module_kwargs (dict, optional): Radial module keyword arguments. Defaults to None.
            shared_radial_module (bool, optional): Whether to use a shared radial module. Defaults to False.
            convert_radial_to_axial (bool, optional): Whether to convert radial to axial. Defaults to False.
            axial_from_radial_kwargs (dict, optional): Keyword arguments for converting radial to axial. Defaults to None.
            spatial_dim (int, optional): Spatial dimension. Defaults to 3.
        """
        super().__init__()

        assert TensorReps("1x1").dim == spatial_dim, "spatial dim must match with tensor rep"
        self.spatial_dim = spatial_dim
        list_in_reps = [parse_reps(reps) for reps in list_in_reps]  # parse from str if necessary
        self.list_reps = list_in_reps + [parse_reps(sam_out_reps)]
        self.global_sam = global_sam
        self.cache_layer_outputs = cache_layer_outputs
        self.list_r = list_r
        self.use_dynamic_sam = use_dynamic_sam

        if use_dynamic_sam:
            assert list_dynamic_k is not None, "list_dynamic_k must be provided for dynamic sam"
        else:
            assert list_r is not None, "list_r must be provided for sam"

        # parse kwargs lists:
        list_conv_kwargs = {} if list_conv_kwargs is None else list_conv_kwargs
        list_sam_kwargs = {} if list_sam_kwargs is None else list_sam_kwargs
        list_conv_kwargs = repeat_in_list(list_conv_kwargs, repeats=len(list_hidden_channels))
        list_sam_kwargs = repeat_in_list(list_sam_kwargs, repeats=len(list_hidden_channels))
        list_lframes_update_kwargs = repeat_in_list(
            list_lframes_update_kwargs, repeats=len(list_hidden_channels)
        )

        lists_for_consistency_check = [
            list_hidden_channels,
            list_in_reps,
            list_dynamic_k if use_dynamic_sam else list_r,
            list_center_sampler,
            list_conv_kwargs,
            list_sam_kwargs,
            list_lframes_update_kwargs,
        ]
        if list_second_hidden_channels is not None:
            lists_for_consistency_check.append(list_second_hidden_channels)
        self.num_sam_layers = consistent_length_check(lists_for_consistency_check)

        # init radial modules:
        list_radial_modules, shared_radial_module = self.build_radial_modules(
            radial_module_type=radial_module_type,
            radial_module_kwargs=radial_module_kwargs,
            adjust_max_radius_with_sampling=False
            if self.list_r is None
            else adjust_max_radius_with_sampling,
            radial_module_shared=shared_radial_module,
            convert_radial_to_axial=convert_radial_to_axial,
            axial_from_radial_kwargs=axial_from_radial_kwargs,
        )

        # init module list
        self.sa_modules = torch.nn.ModuleList()
        for i, hidden_channels in enumerate(list_hidden_channels):
            if shared_radial_module:
                radial_module = shared_radial_module
            else:
                radial_module = list_radial_modules[i]

            second_hidden_channels = (
                list_second_hidden_channels[i] if list_second_hidden_channels else None
            )

            # hidden channel magic has to happen here such that the output can match the next input.
            if i == 0:
                lframes_learner_i = lframes_learner
                lframes_updater_i = None

                conv_kwargs = list_conv_kwargs[i].copy()
                if first_concatenate_edge_vec:
                    conv_kwargs["concatenate_edge_vec"] = True

                sam_kwargs = list_sam_kwargs[i].copy()
                if first_concats_pos_to_features:
                    sam_kwargs["concat_pos_to_features"] = True
            else:
                lframes_learner_i = None
                lframes_updater_kwargs = list_lframes_update_kwargs[i]
                if lframes_updater_kwargs is None:
                    lframes_updater_i = None
                else:
                    lframes_updater_i = QuaternionsUpdateLFrames(
                        in_reps=list_in_reps[i],
                        **list_lframes_update_kwargs[i],
                    )
                conv_kwargs = list_conv_kwargs[i]
                sam_kwargs = list_sam_kwargs[i]

            conv = EdgeConv(
                in_reps=list_in_reps[i],
                hidden_channels=hidden_channels,
                second_hidden_channels=second_hidden_channels,
                out_channels=self.list_reps[i + 1].dim,
                spatial_dim=self.spatial_dim,
                radial_module=radial_module,
                **conv_kwargs,
            )

            if use_dynamic_sam:
                self.sa_modules.append(
                    DynamicSAModule(
                        k=list_dynamic_k[i],
                        conv=conv,
                        center_sampler=list_center_sampler[i],
                        lframes_learner=lframes_learner_i,
                        lframes_updater=lframes_updater_i,
                        **sam_kwargs,
                    )
                )
            else:
                self.sa_modules.append(
                    SAModule(
                        r=list_r[i],
                        conv=conv,
                        center_sampler=list_center_sampler[i],
                        lframes_learner=lframes_learner_i,
                        lframes_updater=lframes_updater_i,
                        **list_sam_kwargs[i],
                    )
                )

    def build_radial_modules(
        self,
        radial_module_type: str,
        radial_module_kwargs: dict,
        adjust_max_radius_with_sampling: bool,
        radial_module_shared: bool,
        convert_radial_to_axial: bool,
        axial_from_radial_kwargs: dict,
    ) -> tuple[list[torch.nn.Module], torch.nn.Module | None]:
        """Builds the radial modules based on the specified parameters.

        Args:
            radial_module_type (str): The type of radial module to use.
            radial_module_kwargs (dict): Additional keyword arguments for the radial module.
            adjust_max_radius_with_sampling (bool): Whether to adjust the maximum radius with sampling.
            radial_module_shared (bool): Whether the radial module is shared across layers.
            convert_radial_to_axial (bool): Whether to convert the radial module to an axial module.
            axial_from_radial_kwargs (dict): Additional keyword arguments for the axial module.

        Returns:
            tuple[list[torch.nn.Module], torch.nn.Module | None]: A tuple containing a list of radial modules and a shared radial module (if applicable).
        """
        if radial_module_type == "none":
            return [None] * self.num_sam_layers, None

        radial_module_kwargs = {} if radial_module_kwargs is None else radial_module_kwargs
        axial_from_radial_kwargs = (
            {} if axial_from_radial_kwargs is None else axial_from_radial_kwargs
        )

        list_radial_modules = []
        if radial_module_shared:
            if radial_module_type == "gaussian":
                radial_module = GaussianEmbedding(**radial_module_kwargs)
            else:
                raise ValueError(f"radial_module_type {radial_module_type} not supported")

            if convert_radial_to_axial:
                radial_module = AxisWiseEmbeddingFromRadial(
                    radial_embedding=radial_module, **axial_from_radial_kwargs
                )

            return list_radial_modules, radial_module

        else:
            for i in range(self.num_sam_layers):
                if radial_module_type == "gaussian":
                    if adjust_max_radius_with_sampling:
                        radial_module_kwargs["maximum_initial_range"] = self.list_r[i]
                        if convert_radial_to_axial:
                            radial_module_kwargs["minimum_initial_range"] = -self.list_r[i]
                    radial_module = GaussianEmbedding(**radial_module_kwargs)
                else:
                    raise ValueError(f"radial_module_type {radial_module_type} not supported")

                if convert_radial_to_axial:
                    radial_module = AxisWiseEmbeddingFromRadial(
                        radial_embedding=radial_module, **axial_from_radial_kwargs
                    )

                list_radial_modules.append(radial_module)
        return list_radial_modules, None

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        lframes: LFrames,
        epoch: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, LFrames, list[dict]]:
        """Forward pass of the PointNet Encoder model.

        Args:
            x (torch.Tensor): Input tensor.
            pos (torch.Tensor): Position tensor.
            batch (torch.Tensor): Batch tensor.
            lframes (LFrames): LFrames object.
            epoch (int | None, optional): Current epoch. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, LFrames, list[dict]]: Output tensors and cached layer outputs.
        """
        cached_layer_outputs = (
            [dict(x=x, pos=pos, batch=batch, lframes=lframes)]
            if self.cache_layer_outputs
            else None
        )

        # sam layers:
        for sam in self.sa_modules:
            x, pos, batch, lframes = sam(x=x, pos=pos, batch=batch, lframes=lframes, epoch=epoch)
            if self.cache_layer_outputs:
                cached_layer_outputs.append(dict(x=x, pos=pos, batch=batch, lframes=lframes))

        if self.global_sam is not None:
            x, pos, batch, lframes = self.global_sam(
                x=x,
                pos=pos,
                batch=batch,
                lframes=lframes,
                cached_layer_outputs=cached_layer_outputs,
            )
            if self.cache_layer_outputs:
                cached_layer_outputs.append(dict(x=x, pos=pos, batch=batch, lframes=lframes))

        return x, pos, batch, lframes, cached_layer_outputs


class PointNetDecoder(torch.nn.Module):
    """Decoder in a PointNet architecture."""

    def __init__(
        self,
        encoder_module: PointNetEncoder,
        list_in_reps: list[Union[TensorReps, Irreps, str]],
        list_hidden_channels: list[list[int]],
        out_reps: Union[TensorReps, Irreps, str],
        list_k: Union[int, list[int]] = 3,
        list_mlp_kwargs: Union[dict, list[dict]] | None = None,
        list_lframes_update_kwargs: Union[dict, list[dict]] | None = None,
    ) -> None:
        """Initializes the PointNetDecoder module.

        Args:
            encoder_module (PointNetEncoder): The encoder module used in the PointNet architecture.
            list_in_reps (list[Union[TensorReps, Irreps, str]]): A list of input representations for each layer of the decoder.
            list_hidden_channels (list[list[int]]): A list of hidden channel sizes for each layer of the decoder.
            out_reps (Union[TensorReps, Irreps, str]): The output representations of the decoder.
            list_k (Union[int, list[int]], optional): The number of nearest neighbors to consider for each layer of the decoder. Defaults to 3.
            list_mlp_kwargs (Union[dict, list[dict]] | None, optional): Additional keyword arguments for the MLP module in each layer of the decoder. Defaults to None.
            list_lframes_update_kwargs (Union[dict, list[dict]] | None, optional): Additional keyword arguments for the UpdateLFramesModule in each layer of the decoder. Defaults to None.

        Raises:
            AssertionError: If the encoder module has a global sam module when used with the decoder.
        """
        super().__init__()
        assert (
            encoder_module.global_sam is None
        ), "encoder should not have a global sam module when used with decoder"
        list_in_reps = [parse_reps(reps) for reps in list_in_reps]  # parse from str if necessary
        self.list_reps = list_in_reps + [parse_reps(out_reps)]
        self.cache_layer_outputs = encoder_module.cache_layer_outputs

        list_k = repeat_in_list(list_k, repeats=len(list_hidden_channels))
        list_mlp_kwargs = {} if list_mlp_kwargs is None else list_mlp_kwargs
        list_mlp_kwargs = repeat_in_list(list_mlp_kwargs, repeats=len(list_hidden_channels))
        list_lframes_update_kwargs = repeat_in_list(
            list_lframes_update_kwargs, repeats=len(list_hidden_channels)
        )

        consistent_length_check(
            [
                list_hidden_channels,
                list_in_reps,
                list_mlp_kwargs,
                list_k,
                list_lframes_update_kwargs,
                encoder_module.sa_modules,
            ]
        )

        # init module list
        self.fp_modules = torch.nn.ModuleList()
        assert list_in_reps[0] == encoder_module.list_reps[-1], "in_reps must match encoder out"
        for i, hidden_channels in enumerate(list_hidden_channels):
            in_reps = list_in_reps[i]
            skip_reps = encoder_module.list_reps[-i - 2]
            hidden_channels.append(self.list_reps[i + 1].dim)
            mlp = MLPWrapped(
                in_channels=in_reps.dim + skip_reps.dim,
                hidden_channels=hidden_channels,
                **list_mlp_kwargs[i],
            )
            if list_lframes_update_kwargs[i] is None:
                lframes_updater = None
            else:
                lframes_updater = QuaternionsUpdateLFrames(
                    in_reps=in_reps + skip_reps,
                    **list_lframes_update_kwargs[i],
                )
            self.fp_modules.append(
                FPModule(
                    mlp=mlp,
                    reps=in_reps,
                    k=list_k[i],
                    lframes_updater=lframes_updater,
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        batch: torch.Tensor,
        lframes: LFrames,
        cached_layer_outputs: list[dict],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, LFrames, list[dict]]:
        """Forward pass of the PointNet module.

        Args:
            x (torch.Tensor): Input tensor.
            pos (torch.Tensor): Position tensor.
            batch (torch.Tensor): Batch tensor.
            lframes (LFrames): LFrames object.
            cached_layer_outputs (list[dict]): List of dictionaries containing cached layer outputs.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, LFrames, list[dict]]: Tuple containing the output tensors and updated cached layer outputs.
        """
        dec_cached_layer_outputs = [] if self.cache_layer_outputs else None
        for i, fp in enumerate(self.fp_modules):
            x_skip, pos_skip, batch_skip, lframes_skip = (
                cached_layer_outputs[-i - 2][key] for key in ["x", "pos", "batch", "lframes"]
            )

            x, pos, batch, lframes = fp(
                x, pos, batch, lframes, x_skip, pos_skip, batch_skip, lframes_skip
            )
            if self.cache_layer_outputs:
                dec_cached_layer_outputs.append(dict(x=x, pos=pos, batch=batch, lframes=lframes))
        return x, pos, batch, lframes, dec_cached_layer_outputs


class PointNet(torch.nn.Module):
    """PointNet model implementation."""

    def __init__(
        self,
        estimate_lframes_module: torch.nn.Module,
        pointnetpp_encoder: PointNetEncoder,
        from_local_to_global_frame: FromLocalToGlobalFrame,
        pointnetpp_decoder: PointNetDecoder | None = None,
        from_global_to_local_frame: FromGlobalToLocalFrame | None = None,
        final_lframes_layer: FinalLframesLayer | None = None,
    ) -> None:
        """Initializes the PointNet model.

        Args:
            estimate_lframes_module (torch.nn.Module): Module for estimating local frames.
            from_global_to_local_frame (FromGlobalToLocalFrame): Transformation from global to local frame.
            from_local_to_global_frame (FromLocalToGlobalFrame): Transformation from local to global frame.
            pointnetpp_encoder (PointNetEncoder): PointNet++ encoder.
            pointnetpp_decoder (PointNetDecoder | None, optional): PointNet++ decoder. Defaults to None.
            final_lframes_layer (FinalLframesLayer | None, optional): Final layer for local frames. Defaults to None.
            output_statistics_layer (StatisticsLayer | None, optional): Layer for output statistics. Defaults to None.
        """
        super().__init__()
        self.estimate_lframes_module = estimate_lframes_module
        self.pointnetpp_encoder = pointnetpp_encoder
        self.pointnetpp_decoder = pointnetpp_decoder
        self.final_lframes_layer = final_lframes_layer
        self.from_global_to_local_frame = from_global_to_local_frame
        self.from_local_to_global_frame = from_local_to_global_frame

        self.cache_layer_outputs = pointnetpp_encoder.cache_layer_outputs
        if pointnetpp_decoder is not None:
            self.pointnetpp_encoder.cache_layer_outputs = True  # needed for skip connections

    def forward(self, data, return_cached_layer_outputs=False, epoch=None):
        """Forward pass of the PointNet model.

        Args:
            data (torch_geometric.data.Data): Input data.
            return_cached_layer_outputs (bool, optional): Whether to return cached layer outputs. Defaults to False.
            epoch (int, optional): Current epoch. Defaults to None.

        Returns:
            torch.Tensor: Output tensor.
            dict: Dictionary containing cached layer outputs (if return_cached_layer_outputs is True).
        """
        x = data.x
        pos = data.pos
        batch = data.batch

        if isinstance(self.estimate_lframes_module, WrappedLearnedLFrames):
            x, lframes = self.estimate_lframes_module(x=x, pos=pos, batch=batch)
        else:
            lframes = self.estimate_lframes_module(pos=pos, batch=batch)

        if self.from_global_to_local_frame is not None:
            x = self.from_global_to_local_frame(x, lframes=lframes)

        x, pos, batch, lframes, enc_cached_layer_outputs = self.pointnetpp_encoder(
            x, pos, batch, lframes, epoch=epoch
        )

        if self.pointnetpp_decoder is None:
            dec_cached_layer_outputs = None
        else:
            x, pos, batch, lframes, dec_cached_layer_outputs = self.pointnetpp_decoder(
                x, pos, batch, lframes, enc_cached_layer_outputs
            )

        if self.final_lframes_layer is not None:
            x, pos, batch, lframes = self.final_lframes_layer(x, pos, batch, lframes)

        x = self.from_local_to_global_frame(x, lframes=lframes)

        if return_cached_layer_outputs:
            return x, dict(
                enc_cached_layer_outputs=enc_cached_layer_outputs,
                dec_cached_layer_outputs=dec_cached_layer_outputs,
            )
        else:
            return x

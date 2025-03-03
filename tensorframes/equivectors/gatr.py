import torch

from tensorframes.equivectors.base import EquiVectors
from experiments.baselines.gatr import GATr, SelfAttentionConfig, MLPConfig
from experiments.baselines.gatr.interface import embed_vector, extract_vector


class GATrWrapper(EquiVectors):
    """
    GATr model to predict vectors based on scalars and vectors
    """

    def __init__(
        self,
        n_vectors,
        in_nodes,
        hidden_mv_channels,
        hidden_s_channels,
        num_blocks,
        num_heads,
        multi_query=False,
        increase_hidden_channels=2,
        head_scale=False,
        double_layernorm=False,
        activation="gelu",
        dropout_prob=None,
    ):
        super().__init__()

        attention = SelfAttentionConfig(
            multi_query=multi_query,
            num_heads=num_heads,
            increase_hidden_channels=increase_hidden_channels,
            dropout_prob=dropout_prob,
            head_scale=head_scale,
        )
        mlp = MLPConfig(
            activation=activation,
            dropout_prob=dropout_prob,
        )
        self.net = GATr(
            in_mv_channels=1,
            out_mv_channels=n_vectors,
            in_s_channels=in_nodes,
            out_s_channels=None,
            hidden_mv_channels=hidden_mv_channels,
            hidden_s_channels=hidden_s_channels,
            num_blocks=num_blocks,
            attention=attention,
            mlp=mlp,
            double_layernorm=double_layernorm,
        )

    def forward(self, fourmomenta, scalars, attn_mask=None):
        # TODO: reshaping business

        mv = embed_vector(fourmomenta)
        s = scalars

        output_mv, _ = self.net(mv, s, attention_mask=attn_mask)
        vecs = extract_vector(output_mv)
        return vecs

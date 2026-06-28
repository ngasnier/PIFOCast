import jax
import jax.numpy as jnp
import flax.linen as nn
from .mlputils import MLP


class PifoModel(nn.Module):
    """Single-graph GNN model. Operates on (NNodes, 6) node features.
    For batching, use jax.vmap."""
    latent_size: int = 8
    mlp_hidden_size: int = 8
    num_message_passing_steps: int = 4
    num_mlp_hidden_layers: int = 1
    output_size: int = 3
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, node_features, edge_features, sources, targets):
        sources = jnp.array(sources)
        targets = jnp.array(targets)

        # Initial mapping: input features -> hidden state
        nodes = nn.Dense(8, kernel_init=nn.initializers.variance_scaling(
            1.0, "fan_in", "truncated_normal"), name="init_node_dense")(node_features)
        nodes = nn.relu(nodes)

        edges = nn.Dense(4, kernel_init=nn.initializers.variance_scaling(
            1.0, "fan_in", "truncated_normal"), name="init_edge_dense")(edge_features)
        edges = nn.relu(edges)

        input_state = node_features[:, :3]

        # Encoder
        nodes = MLP(num_hidden_layers=self.num_mlp_hidden_layers,
                     hidden_size=self.mlp_hidden_size,
                     output_size=self.latent_size,
                     use_layer_norm=self.use_layer_norm,
                     name="encoder/node")(nodes)
        edges = MLP(num_hidden_layers=self.num_mlp_hidden_layers,
                     hidden_size=self.mlp_hidden_size,
                     output_size=self.latent_size,
                     use_layer_norm=self.use_layer_norm,
                     name="encoder/edge")(edges)

        # Processor
        for step in range(self.num_message_passing_steps):
            src = nodes[sources]
            tgt = nodes[targets]
            edge_input = jnp.concatenate([edges, src, tgt], axis=-1)
            edge_update = MLP(num_hidden_layers=self.num_mlp_hidden_layers,
                               hidden_size=self.mlp_hidden_size,
                               output_size=self.latent_size,
                               use_layer_norm=self.use_layer_norm,
                               name=f"processor_{step}/edges")(edge_input)

            messages = jax.ops.segment_sum(edge_update, targets, num_segments=nodes.shape[0])
            node_input = jnp.concatenate([nodes, messages], axis=-1)
            node_update = MLP(num_hidden_layers=self.num_mlp_hidden_layers,
                               hidden_size=self.mlp_hidden_size,
                               output_size=self.latent_size,
                               use_layer_norm=self.use_layer_norm,
                               name=f"processor_{step}/nodes")(node_input)

            nodes = nodes + node_update
            edges = edges + edge_update

        # Decoder
        decoded = MLP(num_hidden_layers=self.num_mlp_hidden_layers,
                       hidden_size=self.mlp_hidden_size,
                       output_size=self.output_size,
                       use_layer_norm=False,
                       name="decoder")(nodes)

        # Prediction head
        delta = MLP(num_hidden_layers=self.num_mlp_hidden_layers,
                     hidden_size=self.mlp_hidden_size,
                     output_size=self.output_size,
                     use_layer_norm=False,
                     activate_final=False,
                     name="prediction_head")(decoded)

        return input_state + delta

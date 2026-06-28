import jax
import jax.numpy as jnp
import flax.linen as nn

class MLP(nn.Module):
    num_hidden_layers: int
    hidden_size: int
    output_size: int
    use_layer_norm: bool
    activation: str = "silu"
    activate_final: bool = False

    @nn.compact
    def __call__(self, x):
        for i in range(self.num_hidden_layers):
            x = nn.Dense(self.hidden_size, kernel_init=nn.initializers.variance_scaling(
                1.0, "fan_in", "truncated_normal"), name=f"dense_{i}")(x)
            x = getattr(nn, self.activation)(x)
        x = nn.Dense(self.output_size, kernel_init=nn.initializers.variance_scaling(
            1.0, "fan_in", "truncated_normal"), name=f"dense_{self.num_hidden_layers}")(x)
        if self.activate_final:
            x = getattr(nn, self.activation)(x)
        if self.use_layer_norm:
            x = nn.LayerNorm(name="layer_norm")(x)
        return x


def build_mlp(num_hidden_layers: int,
              hidden_size: int,
              output_size: int,
              use_layer_norm: bool,
              activation: str = "silu",
              activate_final: bool = False,
              name: str = "mlp") -> MLP:
    return MLP(num_hidden_layers=num_hidden_layers,
               hidden_size=hidden_size,
               output_size=output_size,
               use_layer_norm=use_layer_norm,
               activation=activation,
               activate_final=activate_final,
               name=name)

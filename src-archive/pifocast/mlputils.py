import tensorflow as tf

# Note that gradients vanishes with relu, leaky_relu is more robust but the silu function seems to work best,  
# it is used in Gencast after all. So we default to that.

def build_mlp(num_hidden_layers: int,
              hidden_size: int,
              output_size: int,
              use_layer_norm: bool,
              activation: str = "silu",#tf.keras.layers.LeakyReLU(alpha=0.01), 
              activate_final: bool = False,
              name: str = "mlp"):
    """Builds an MLP."""
    output_sizes = [hidden_size] * num_hidden_layers + [output_size]
    mlp = tf.keras.Sequential(name="mlp")
    for layer_i, size in enumerate(output_sizes):
        layer_activation = activation
        if not activate_final and layer_i == len(output_sizes) - 1:
            layer_activation = None
        mlp.add(tf.keras.layers.Dense(
            size,
            activation=layer_activation,
            use_bias=True,
            kernel_initializer="variance_scaling",
            bias_initializer="zeros",
            name=f"{name}/dense_{layer_i}"))
    if use_layer_norm:
        mlp.add(tf.keras.layers.LayerNormalization(
            name=f"{name}/layer_norm"))
    return mlp
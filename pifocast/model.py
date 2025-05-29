import tensorflow as tf
import tensorflow_gnn as tfgnn
from .LatLonGrid import LatLonGrid
from .PifoEncodeProcessDecode import PifoEncodeProcessDecode
from .mlputils import build_mlp

def buildGridGNN(X, X_edges, grid:LatLonGrid, **task_kwargs):
    """
    Parameters : 
    X : [NLat x NLon, NFeatures], the input data 
    Y : [NLat x NLon, NFeatures], the expected output data (for learning)
    latitudes : array of NLat latitudes
    longitudes : array of NLon longitudes
    """
   
    graph = tfgnn.GraphTensor.from_pieces(
        node_sets={
            "grid": tfgnn.NodeSet.from_fields(
                sizes=tf.constant([ grid.NNodes ]),
                features={
                    "features": X
                }
            )
        },
        edge_sets={
            "edges": tfgnn.EdgeSet.from_fields(
                sizes=tf.constant([ grid.NEdges ]),
                features={
                    "edge_features": X_edges
                },
                adjacency=tfgnn.Adjacency.from_indices(
                    source=("grid", tf.constant(grid.sources)),
                    target=("grid", tf.constant(grid.targets))
                )
            )
        }
    )

    return graph


denseMapping = tf.keras.layers.Dense(8, "relu")
edge_denseMapping = tf.keras.layers.Dense(4, "relu")
def _set_initial_node_state(node_set, *, node_set_name):
    assert node_set_name == "grid"
    return denseMapping(node_set["features"])
#tf.cast(node_set["features"], tf.float32)
    #
    #return tf.concat(
#        [tf.cast(node_set["features"], tf.float32)[..., None],
#        ],
#        axis=-1)

def _set_initial_edge_state(edge_set, *, edge_set_name):
    assert edge_set_name == "edges"
    return edge_denseMapping(edge_set["edge_features"])
    #return tfgnn.keras.layers.MakeEmptyFeature()(edge_set)

def _set_initial_context_state(context):
    return tfgnn.keras.layers.MakeEmptyFeature()(context)

build_initial_hidden_state = tfgnn.keras.layers.MapFeatures(
    node_sets_fn=_set_initial_node_state,
    edge_sets_fn=_set_initial_edge_state,
    context_fn=_set_initial_context_state)


def pifo(gtspec: tfgnn.GraphTensorSpec, 
        num_message_passing_steps=1,
        num_mlp_hidden_layers=1,
        mlp_hidden_size=8, 
        latent_size=8,
        output_size=3):
    
    # Input graph from dataset or inference with all input features
    input_graph = tf.keras.layers.Input(type_spec=gtspec)

    # We want to output a difference. So keep track of input features.
    input_state = tfgnn.keras.layers.Readout(
                node_set_name="grid",
                feature_name="features"
            )(input_graph)[:,0:3]

    # Map input features to hidden state
    output_graph = build_initial_hidden_state(input_graph)

    # Apply the encode process decode algorithm
    trainable_gnn = PifoEncodeProcessDecode(
        edge_output_size=None, # No feature output on edge
        node_output_size=output_size,  # Only one feature to predict for now
        context_output_size=None,  # Don't need this output.
        # Other configurable hyperparameters (most combinations should train).
        num_message_passing_steps=num_message_passing_steps,
        num_mlp_hidden_layers=num_mlp_hidden_layers,
        mlp_hidden_size=mlp_hidden_size ,
        latent_size=latent_size,      # taille des features sur chaque noeud
        use_layer_norm=True,
        shared_processors=False,
        )

    output_graph = trainable_gnn(output_graph)

    # Readout final features [num nodes, n features]
    output = tfgnn.keras.layers.Readout(
                node_set_name="grid",
                feature_name=tfgnn.HIDDEN_STATE
            )(output_graph)
    
    # Apply a prediction head to get Y(t) the difference from X(t) to new state
    output = build_mlp(num_hidden_layers=num_mlp_hidden_layers,
        hidden_size=mlp_hidden_size,
        output_size=output_size,
        activate_final=False,
        use_layer_norm=False,  
        name=f"pifo/prediction_head")(output)
    
    # Now we want our output to be X(t) + Y(t)
    output = tf.math.add(input_state, output)
    
    # Now we have a trainable/savable Keras model
    model = tf.keras.Model(input_graph, output)

    return model



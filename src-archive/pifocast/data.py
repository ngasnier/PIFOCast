import math
import xarray as xr
import tensorflow as tf
import tensorflow_gnn as tfgnn
from .LatLonGrid import LatLonGrid
from .model import buildGridGNN

def normalizeField(field):
    min = tf.math.reduce_min(field)
    max = tf.math.reduce_max(field)
    return tf.math.divide(
            tf.math.subtract(
                field, 
                min
            ), 
            tf.math.subtract(
                max, 
                min
            )
        ), min, max


def pifoGridGenerator(grib_file: str, 
                      grid:LatLonGrid,
                      stride:int = 4,
                      step:int = 1,
                      **task_kwargs):
    """
    Generator of graph and labels for training.
    Parameters : 
    grib_file : a ERA5 grib file path, with Z, U and V fields at 500hPa.
    grid : a LatLonGrid with corresponding shape.
    stride : default 4. To reduce resolution of the grib data.
    step : default 1. Defines how to advance in grib file to generate examples.
    Yields :
    GraphTensor of pifo.pbtxt schema, with corresponding normalized features on nodes and edges
    Tensor concatenating Z, U, V normalized features of prediction, according to the LatLonGrid parameters.
    """
    # Load data (8 z500 ggrib_file...)
    ds = xr.load_dataset(grib_file, engine="cfgrib", decode_timedelta=False)
    # Sélection d'une grille réduite avec un sous-échantillonnage (1 point sur 4)
    ds_reduced = ds.isel(latitude=slice(0, None, stride), longitude=slice(0, None, stride))

    G_NLat = ds_reduced.latitude.shape[0]
    G_NLon = ds_reduced.longitude.shape[0]
    assert G_NLat==grid.NLat and G_NLon==grid.NLon, "grid definition and grib file mismatch."

    NExamples = ds_reduced.z.shape[0]

    # Create train data
    Z = tf.reshape(tf.convert_to_tensor(ds_reduced.z[0:NExamples]), [NExamples, grid.NNodes])
    Z, _ , _ = normalizeField(Z) # Z/70000 # normalisation à l'arrache
    m = Z.shape[0]

    u = tf.reshape(tf.convert_to_tensor(ds_reduced.u[0:NExamples]), [NExamples, grid.NNodes])
    u, _, _ = normalizeField(u)
    v = tf.reshape(tf.convert_to_tensor(ds_reduced.v[0:NExamples]), [NExamples, grid.NNodes])
    v, _, _ = normalizeField(v)

    for i in range(0, m-1, step): # To allow for making Y
        print("    grib index ", i)
        features = tf.stack([Z[i], u[i], v[i]], axis=1)  # , cos_lat, cos_lon, sin_lon
        predictions = tf.stack([Z[i+1], u[i+1], v[i+1]], axis=1)        
        yield getGraphForFeatures(features, grid), predictions  #  buildGridGNN(features, edgeFeatures, grid)


def getGraphExample(grid:LatLonGrid, file="dataset/202504.grib"):
    """
    Get an example of graph from a file.
    """
    generator = pifoGridGenerator(file, grid)
    return next(generator)


# Quick & dirty way of chaining predictions with ...
def getGraphForFeatures(X, grid:LatLonGrid):
    #edgeFeatures = tf.zeros([grid.NEdges, NEdgeFeatures], dtype=tf.float32)
    edge_lengthes = tf.convert_to_tensor(grid.lengthes, dtype_hint="float")
    edge_lengthes, _, _  = normalizeField(edge_lengthes)
    x, _, _ = normalizeField(tf.convert_to_tensor(grid.coords_x, dtype_hint="float"))
    y, _, _ = normalizeField(tf.convert_to_tensor(grid.coords_y, dtype_hint="float"))
    edgeFeatures = tf.stack([
            edge_lengthes, 
            x,
            y
        ], axis=1)


    cos_lat, _, _ = normalizeField(tf.math.sin(grid.lat2d*math.pi/180))
    cos_lon, _, _ = normalizeField(tf.math.cos(grid.lon2d*math.pi/180))
    sin_lon, _, _ = normalizeField(tf.math.sin(grid.lon2d*math.pi/180))

    features = tf.concat([X, tf.reshape(cos_lat, (-1, 1)), tf.reshape(cos_lon, (-1, 1)), tf.reshape(sin_lon, (grid.NNodes, -1))], axis=1)
    return buildGridGNN(features, edgeFeatures, grid)


def get_dataset(grib_file, grid:LatLonGrid) -> tf.data.Dataset:
    def generator_fn():
        return pifoGridGenerator(grib_file, grid)
    
    schema = tfgnn.read_schema("pifo.pbtxt")
    graph_spec = tfgnn.create_graph_spec_from_schema_pb(schema)
    return tf.data.Dataset.from_generator( 
        generator_fn, output_signature=(graph_spec, tf.TensorSpec((grid.NNodes,3), dtype=tf.float32))).repeat()

def load_dataset(graphSpec, graphFile, targetFile, batch_size=1) -> tf.data.Dataset:
    """
    Load a dataset with (GraphTensor, Tensor) elements from TFRecord files.
    Parameters :
    graphSpec: the graph spec for the graph files
    graphFile: path to the TFRecord file containing graph training examples.
    targetFile: path to the TFRecord containing the target output tensor of shape [nnodes, 3]
    batch_size: default 1. The returned dataset will be batched accordingly.
    """
    targetFeature = {
        'target' : tf.io.FixedLenFeature([], tf.string) #tf.io.RaggedFeature(dtype=tf.float32)
    }    

    def mapfn(Y):
        parsed_example = tf.io.parse_single_example(Y, targetFeature)
        target = tf.io.parse_tensor(parsed_example['target'], out_type=tf.float32)
        return target


    graphDS = tf.data.TFRecordDataset(graphFile)
    graphDS = graphDS.map(
        lambda serialized: tfgnn.parse_single_example(graphSpec, serialized))
    input_spec = graphDS.element_spec

    targetDS = tf.data.TFRecordDataset(targetFile)
    targetDS = targetDS.map(mapfn, num_parallel_calls=tf.data.AUTOTUNE)

    trainDS = tf.data.Dataset.zip((graphDS, targetDS))
    trainDS = trainDS.repeat()

    if batch_size>1:
        # Needs to merge targets because of merge_batch_to_components() occuring in the model.
        # The output tensor will not keep the batch dimension, and this will fail the loss function.
        trainDS = trainDS.batch(batch_size, drop_remainder=True)
        def merge_target(target):
            # Reshape [batch, nnodes, 3] -> [batch*nnodes, 3]
            return tf.reshape(target, [-1, 3])

        trainDS = trainDS.map(lambda graph, target: (graph, merge_target(target)))

    return trainDS
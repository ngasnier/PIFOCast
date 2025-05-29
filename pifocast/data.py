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
                      **task_kwargs):
    """
    Generator of graph and labels for training.
    Parameters : 
    grib_file : a ERA5 grib file path, with Z, U and V fields at 500hPa.
    grid : a LatLonGrid with corresponding shape.
    stride : default 4. To reduce resolution of the grib data.
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
    # Nfeatures = 3
    # NEdgeFeatures = 3
  
    # cos_lat, _, _ = normalizeField(tf.math.sin(grid.lat2d*math.pi/180))
    # cos_lon, _, _ = normalizeField(tf.math.cos(grid.lon2d*math.pi/180))
    # sin_lon, _, _ = normalizeField(tf.math.sin(grid.lon2d*math.pi/180))

    # Create train data
    Z = tf.reshape(tf.convert_to_tensor(ds_reduced.z[0:NExamples]), [NExamples, grid.NNodes])
    Z, _ , _ = normalizeField(Z) # Z/70000 # normalisation à l'arrache
    m = Z.shape[0]

    u = tf.reshape(tf.convert_to_tensor(ds_reduced.u[0:NExamples]), [NExamples, grid.NNodes])
    u, _, _ = normalizeField(u)
    v = tf.reshape(tf.convert_to_tensor(ds_reduced.v[0:NExamples]), [NExamples, grid.NNodes])
    v, _, _ = normalizeField(v)

    # edge_lengthes = tf.convert_to_tensor(grid.lengthes, dtype_hint="float")
    # edge_lengthes, _, _ = normalizeField(edge_lengthes)

    # x, _, _ = normalizeField(tf.convert_to_tensor(grid.coords_x, dtype_hint="float"))
    # y, _, _ = normalizeField(tf.convert_to_tensor(grid.coords_y, dtype_hint="float"))
    # edgeFeatures = tf.stack([
    #         edge_lengthes, 
    #         x,
    #         y
    #     ], axis=1)
    

    for i in range(m-1): # To allow for making Y
        features = tf.stack([Z[i], u[i], v[i]], axis=1)  # , cos_lat, cos_lon, sin_lon
        predictions = tf.stack([Z[i+1], u[i+1], v[i+1]], axis=1)        
        yield getGraphForFeatures(features, grid), predictions  #  buildGridGNN(features, edgeFeatures, grid)


def getGraphExample(grid:LatLonGrid, file="dataset/202504.grib"):
    """
    Get an example of graph from a file.
    """
    generator = pifoGridGenerator(file, grid)
    return next(generator)
    #     # Load data (8 z500 ggrib_file...)
    # ds = xr.load_dataset("dataset/202504.grib", engine="cfgrib", decode_timedelta=False)
    # #ds_reduced = ds
    # # Sélection d'une grille réduite avec un sous-échantillonnage (1 point sur 4)
    # ds_reduced = ds.isel(latitude=slice(0, None, 4), longitude=slice(0, None, 4))

    # G_NLat = ds_reduced.latitude.shape[0]
    # G_NLon = ds_reduced.longitude.shape[0]
    # assert G_NLat==grid.NLat and G_NLon==grid.NLon, "grid definition and grib file mismatch."

    # NEdgeFeatures = 3

    # # Create train data
    # NExamples = ds_reduced.z.shape[0]

    # Z = tf.reshape(tf.convert_to_tensor(ds_reduced.z[0:NExamples-1]), [NExamples-1, grid.NNodes])
    # Z, _ , _ = normalizeField(Z)

    # u = tf.reshape(tf.convert_to_tensor(ds_reduced.u[0:NExamples]), [NExamples, grid.NNodes])
    # u, _, _ = normalizeField(u)
    # v = tf.reshape(tf.convert_to_tensor(ds_reduced.v[0:NExamples]), [NExamples, grid.NNodes])
    # v, _, _ = normalizeField(v)

    # cos_lat, _, _ = normalizeField(tf.math.sin(grid.lat2d*math.pi/180))
    # cos_lon, _, _ = normalizeField(tf.math.cos(grid.lon2d*math.pi/180))
    # sin_lon, _, _ = normalizeField(tf.math.sin(grid.lon2d*math.pi/180))

    
    # features = tf.stack([Z[0], u[0], v[0], cos_lat, cos_lon, sin_lon], axis=1)

    # #Nfeatures = 3
    # # edgeFeatures = tf.zeros([grid.NEdges, NEdgeFeatures], dtype=tf.float32)
    # edge_lengthes = tf.convert_to_tensor(grid.lengthes, dtype_hint="float")
    # edge_lengthes, _, _ =  normalizeField(edge_lengthes)
    # x, _, _ = normalizeField(tf.convert_to_tensor(grid.coords_x, dtype_hint="float"))
    # y, _, _ = normalizeField(tf.convert_to_tensor(grid.coords_y, dtype_hint="float"))
    
    # edgeFeatures = tf.stack([
    #         edge_lengthes, 
    #         x,
    #         y
    #     ], axis=1)

    # # for i in range(10):
    # #     ds_reduced.z[i].plot()
    # #     plt.savefig("X_"+str(i)+".png", dpi=150, bbox_inches="tight")  # Ajuste la qualité et l'encadrement
    # #     plt.close()  
    # # ds_reduced.z[1].plot()
    
    # return buildGridGNN(features, edgeFeatures, grid), tf.stack([Z[0], u[0], v[0]])


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

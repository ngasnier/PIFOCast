import math
import numpy as np
import xarray as xr
from .LatLonGrid import LatLonGrid


def normalizeField(field):
    f_min = np.min(field)
    f_max = np.max(field)
    return (field - f_min) / (f_max - f_min), f_min, f_max


def pifoGridGenerator(grib_file: str,
                      grid: LatLonGrid,
                      stride: int = 4,
                      step: int = 1):
    ds = xr.load_dataset(grib_file, engine="cfgrib", decode_timedelta=False)
    ds_reduced = ds.isel(latitude=slice(0, None, stride), longitude=slice(0, None, stride))

    G_NLat = ds_reduced.latitude.shape[0]
    G_NLon = ds_reduced.longitude.shape[0]
    assert G_NLat == grid.NLat and G_NLon == grid.NLon, "grid definition and grib file mismatch."

    NExamples = ds_reduced.z.shape[0]

    Z = np.reshape(np.array(ds_reduced.z[0:NExamples]), [NExamples, grid.NNodes]).astype(np.float32)
    Z, _, _ = normalizeField(Z)
    u = np.reshape(np.array(ds_reduced.u[0:NExamples]), [NExamples, grid.NNodes]).astype(np.float32)
    u, _, _ = normalizeField(u)
    v = np.reshape(np.array(ds_reduced.v[0:NExamples]), [NExamples, grid.NNodes]).astype(np.float32)
    v, _, _ = normalizeField(v)

    # Pre-compute edge features (static for the grid)
    edge_features = create_edge_features(grid)

    # Pre-compute position features (static for the grid)
    pos_features = create_position_features(grid)

    sources = np.array(grid.sources, dtype=np.int32)
    targets = np.array(grid.targets, dtype=np.int32)

    for i in range(0, NExamples - 1, step):
        print("    grib index ", i)
        node_features = np.concatenate([
            Z[i:i+1].T, u[i:i+1].T, v[i:i+1].T,
            pos_features
        ], axis=1)
        labels = np.stack([Z[i+1], u[i+1], v[i+1]], axis=1)
        yield node_features, edge_features, sources, targets, labels


def getGraphExample(grid: LatLonGrid, file="dataset/202504.grib"):
    generator = pifoGridGenerator(file, grid)
    return next(generator)


def get_dataset(grib_file, grid: LatLonGrid):
    def generator_fn():
        return pifoGridGenerator(grib_file, grid)
    return generator_fn


def create_edge_features(grid: LatLonGrid):
    edge_lengthes = np.array(grid.lengthes, dtype=np.float32)
    edge_lengthes, _, _ = normalizeField(edge_lengthes)
    cx, _, _ = normalizeField(np.array(grid.coords_x, dtype=np.float32))
    cy, _, _ = normalizeField(np.array(grid.coords_y, dtype=np.float32))
    return np.stack([edge_lengthes, cx, cy], axis=1)


def create_position_features(grid: LatLonGrid):
    cos_lat, _, _ = normalizeField(np.sin(np.array(grid.lat2d) * math.pi / 180))
    cos_lon, _, _ = normalizeField(np.cos(np.array(grid.lon2d) * math.pi / 180))
    sin_lon, _, _ = normalizeField(np.sin(np.array(grid.lon2d) * math.pi / 180))
    return np.column_stack([np.array(cos_lat), np.array(cos_lon), np.array(sin_lon)])

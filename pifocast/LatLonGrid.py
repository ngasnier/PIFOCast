import math
import xarray as xr
import tensorflow as tf

class LatLonGrid:
    def __init__(self, from_grib, regrid_factor=4):
        """
        Parameters :
        from_grib : grib file to take as example of lat lon data
        regrid_factor : grib slicing factor to reduce resolution
        """
        ds = xr.load_dataset(from_grib, engine="cfgrib", decode_timedelta=False)
        ds_reduced = ds.isel(latitude=slice(0, None, regrid_factor), longitude=slice(0, None, regrid_factor))

        self.NLat = ds_reduced.latitude.shape[0]
        self.NLon = ds_reduced.longitude.shape[0]
        self.NNodes = self.NLat * self.NLon

        self.lat2d = tf.reshape(tf.convert_to_tensor(ds_reduced.latitude.expand_dims({"longitude": self.NLon}, axis=1), dtype_hint="float"), [self.NNodes])
        self.lon2d = tf.reshape(tf.convert_to_tensor(ds_reduced.longitude.expand_dims({"latitude": self.NLat}, axis=0), dtype_hint="float"), [self.NNodes])

        self.__buildAdjacency(ds_reduced)

    def __calcEdgeLength(self, ds):
        # Calc length of edges between nodes on meridian between each latitude
        med_length = []
        for j in range(self.NLat-1): 
            med_length.append(
                self.distanceLatLon(float(ds.latitude[j]), 0, 
                                    float(ds.latitude[j+1]),0))
            
        # Calc length of edges between nodes on each latitude circule
        lat_length = []
        for j in range(self.NLat): 
            lat_length.append(
                self.distanceLatLon(float(ds.latitude[j]), float(ds.longitude[0]), 
                                    float(ds.latitude[j]), float(ds.longitude[1])
                )
            )

        return med_length, lat_length


    def __buildAdjacency(self, ds):

        med_length, lat_length = self.__calcEdgeLength(ds)

        sources = []
        targets = []
        lengthes = []
        coords_x = []
        coords_y = []
        # Nodes for poles are note linked around latitude cirles
        node_id = self.NLon # Skip the first row of pole values
        for j in range(1, self.NLat-1): 
            for i in range(self.NLon):                
                # Get data from the right
                if i<self.NLon-1:
                    sources.append(node_id+1)
                    targets.append(node_id)
                    lengthes.append(lat_length[j])
                else:
                    sources.append(node_id-self.NLon+1)
                    targets.append(node_id)
                    lengthes.append(lat_length[j])
                coords_x.append(1)
                coords_y.append(0)

                # Get data from the top (higher lats indexed first)
                sources.append(node_id-self.NLon)
                targets.append(node_id)
                lengthes.append(med_length[j-1])
                coords_x.append(0)
                coords_y.append(1)

                # Get data from the left
                if i==0: # Wrap at greenwich
                    sources.append(node_id+self.NLon-1)
                    targets.append(node_id)
                    lengthes.append(lat_length[j])
                else:
                    sources.append(node_id-1)
                    targets.append(node_id)
                    lengthes.append(lat_length[j])
                coords_x.append(-1)
                coords_y.append(0)

                # Get data from the bottom
                sources.append(node_id+self.NLon)
                targets.append(node_id)
                lengthes.append(med_length[j])
                coords_x.append(0)
                coords_y.append(-1)

                node_id += 1
            
            # prout = len (lengthes)-1
            # print("lat=", float(ds.latitude[j]), " : ")
            # print("  right->[", sources[prout-3], ", ",targets[prout-3],"]", lengthes[prout-3])
            # print("  top<-[", sources[prout-2], ", ",targets[prout-2],"]", lengthes[prout-2])
            # print("  left^^[", sources[prout-1], ", ",targets[prout-1],"]", lengthes[prout-1])
            # print("  bottomvv[", sources[prout], ", ",targets[prout],"]", lengthes[prout])
    

        self.sources = sources
        self.targets = targets
        self.lengthes = lengthes
        self.coords_x = coords_x
        self.coords_y = coords_y
        self.NEdges = len(sources)

    
    
    def distanceLatLon(self, lata, lona, latb, lonb):
        lat1 = lata/180*math.pi
        lon1 = lona/180*math.pi
        lat2 = latb/180*math.pi
        lon2 = lonb/180*math.pi

        d = math.acos(math.sin(lat1)*math.sin(lat2) + math.cos(lat1)*math.cos(lat2)*math.cos(lon2-lon1))
        return d*6371598 # sphere de Picard
                        # 6378137 sphere IAG-GRS80

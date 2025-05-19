import cdsapi

# https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels?tab=overview


#Hersbach, H., Bell, B., Berrisford, P., Biavati, G., Horányi, A., Muñoz Sabater, J., Nicolas, J., Peubey, C., Radu, R., Rozum, I., Schepers, D., Simmons, A., Soci, C., Dee, D., Thépaut, J-N. (2023): ERA5 hourly data on pressure levels from 1940 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS), DOI: 10.24381/cds.bd0915c6 (Accessed on DD-MMM-YYYY)

# $HOME/.cdsapirc :
# url: https://cds.climate.copernicus.eu/api
# key: e57c3cd8-3c0a-4471-9460-b503ea778230

#dataset = "reanalysis-era5-single-levels"
dataset = "reanalysis-era5-pressure-levels"
# request = {
#     "product_type": ["reanalysis"],
#     "variable": ["geopotential"],
#     "year": ["2025"],
#     "month": ["04"],
#     "day": ["06", "07"],
#     "time": [
#         "00:00", "06:00", "12:00",
#         "18:00"
#     ],
#     "pressure_level": ["500"],
#     "data_format": "grib",
#     "download_format": "unarchived"
# }
request = {
    "product_type": ["reanalysis"],
    "variable": [
        "geopotential",
        "u_component_of_wind",
        "v_component_of_wind"
    ],
    "year": ["2025"],
    "month": ["04"],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30"
    ],
    "time": [
        "00:00", "06:00", "12:00",
        "18:00"
    ],
    "pressure_level": ["500"],
    "data_format": "grib",
    "download_format": "unarchived"
}

target = 'dataset/202504.grib'

client = cdsapi.Client()
client.retrieve(dataset, request, target)





client = cdsapi.Client()
client.retrieve(dataset, request).download()

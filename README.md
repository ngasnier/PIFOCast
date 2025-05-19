PIFOCast
========

A deep learning model to do basic global weather forecasts. It is the follow-up to a former javascript project named PIFO,
a basic model framework with a barotropic and a simple baroclinic core. PIFO stands for "Projet Informatique à Formules Ouvertes" 
(Open Formulas Computer Project). This is also a pun word in french that says "at random" or something similar. It is
essentially a hobby project but with serious stuffs in it.

Dependencies
============

The project requires the following to run and setup :
- python : to run all the scripts
- poetry : to setup the venv for the project

The following library are used :
- cdsapi : to retrieve ECMWF ERA 5 data
- xarray : to load and regrid data from ERA5 GRIB files, and processing the datasets
- tensorflow : deep learning framework with CUDA GPU integration
- magics : to plot the weather maps.

Additionnaly, Magics has some system library dependencies, not installed by poetry (examle using Python): 
```
sudo apt install netcdf-bin
sudo apt install proj-bin
```

If you encounter errors like follows, even after installing dependencies :
```
  OSError: libproj.so.15: cannot open shared object file: No such file or directory
```

You need to make symbolic links to lib versions in /usr/lib/x86_64-linux-gnu, e.g :
```
  sudo ln -s libproj.so.25 libproj.so.15
```
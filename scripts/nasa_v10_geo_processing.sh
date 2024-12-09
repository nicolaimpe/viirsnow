#!/bin/bash

PROJ4_MODIS="+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
file='data/V10A1/VNP10A1.A2023002.h07v05.002.2023093165900.h5'

X0=$(ncdump -v XDim $file | sed -n 's/.*XDim = \(.*\),.*/\1/p' | awk -F ',' '{print $1}')
Y0=$(ncdump -v YDim $file | sed -n 's/.*YDim = \(.*\),.*/\1/p' | awk -F ',' '{print $1}')

gdal_edit.py -a_srs  -a_ullr 0 5559752 1111579.86949378 4448172.72884022 ndsi.tif 
# steps to produce a 250m resolution FSC from FSCOG S2 product with a "zombie" nodata mask 
# here target extent is determined by gdal
f=$(find . -name "*FSCOG.tif")
mkdir -p tmp
# create validity mask at 250m resolution
gdalwarp -srcnodata None -r max -tr 250 250 $f tmp/max.tif
# export fsc to float
gdal_translate -ot Float32 $f tmp/fscfloat.tif
# aggregate to 250m using the average 
gdalwarp -srcnodata "205 255" -r average -tr 250 250 -ot Float32 tmp/fscfloat.tif tmp/floatavgfloat.tif
# mask non-valid pixels
gdal_calc.py -A tmp/floatavgfloat.tif -B tmp/max.tif --calc="numpy.where(B>100,255,A)" --NoDataValue=255 --outfile tmp/fscmasked.tif


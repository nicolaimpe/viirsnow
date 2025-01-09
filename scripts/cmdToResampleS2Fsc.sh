#!/bin/sh
# steps to produce a 250m resolution FSC from FSCOG S2 product with a "zombie" nodata mask 
# here target extent is determined by gdal

#VIIRS_FOLDER="/home/imperatoren/work/VIIRS_S2_comparison/viirsnow/output_folder"
#viirs_file=$VIIRS_FOLDER/"WY_2023_2024_SuomiNPP_nasa_time_series.nc"
s2path=$1 #"/home/imperatoren/work/VIIRS_S2_comparison/data/S2/FSC_20231002T105827_S2A_T31TDL_V102_1"
s2path="/home/imperatoren/work/VIIRS_S2_comparison/data/S2/FSC_20170103T104428_S2A_T31TCG_V100_1/"
f=$(find $s2path -name "*FSCOG.tif")

outf=$(basename $f .tif)
mkdir -p $outf
# create validity mask at 250m resolution
# xmin=118644.782
# xmax=1049894.782
# ymin=4616200.982
# ymax=5417950.982

xmin=0
xmax=1050000
ymin=4575000
ymax=5400000
gdalwarp -q -overwrite -t_srs EPSG:32631 -te $xmin $ymin $xmax $ymax -srcnodata None -r max -tr 250 250 $f $outf/max.tif
# export fsc to float
gdal_translate -q -ot Float32 $f $outf/fscfloat.tif
# aggregate to 250m using the average 
gdalwarp -q -overwrite -t_srs EPSG:32631 -te $xmin $ymin $xmax $ymax -srcnodata "205 255" -r average -tr 250 250 -ot Float32 $outf/fscfloat.tif $outf/floatavgfloat.tif
# mask non-valid pixels
gdal_calc.py --co="compress=deflate" --overwrite --quiet -A $outf/floatavgfloat.tif -B $outf/max.tif --calc="numpy.where(B>100,255,A)" --NoDataValue=255 --outfile $outf/fscmasked.tif
# Reconvert in Uint8
gdal_translate -q -ot Byte $outf/fscmasked.tif $outf/s2_fsc_250m.tif
rm $outf/max.tif $outf/floatavgfloat.tif $outf/fscfloat.tif  $outf/*.aux.xml
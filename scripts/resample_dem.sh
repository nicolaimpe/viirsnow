# First I used the MSF 20 m raster and converted to binary mask
# Then I multiplied the mask by the 20 m DEM
# Then apply this for bilinear
gdalwarp -t_srs EPSG:32631 -tr 375 375 -te 0 4575000 1050000 5400000 -srcnodata 0 -r bilinear DEM_MASKED_L93_20m_bilinear.tif DEM_MSF_UTM32_375m_bilinear.tif
# and for lanczos
gdalwarp -t_srs EPSG:32631 -tr 375 375 -te 0 4575000 1050000 5400000 -srcnodata 0 -r lanczos DEM_MASKED_L93_20m_bilinear.tif DEM_MSF_UTM32_375m_lanczos.tif

# Then for slope
gdaldem slope -alg horn DEM_MSF_UTM32_375m_bilinear.tif SLP_MSF_UTM32_375m_bilinear.tif
gdaldem slope -alg horn DEM_MSF_UTM32_375m_lanczos.tif SLP_MSF_UTM32_375m_lanczos.tif

# Foresrt mask
gdalwarp -t_srs EPSG:32631 -tr 500 500 -te 0 4575000 1050000 5400000 -r nearest europe/corine_2006_forest_mask_europe.tif corine_2006_forest_mask_utm_500m.tif
# First I used the MSF 20 m raster and converted to binary mask
# Then I multiplied the mask by the 20 m DEM
# Then apply this for bilinear
gdalwarp -t_srs "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs" -tr 370.650173222222 370.650173222222 -te -420000 4486309.549622223 877275.6062777771 5450000 -srcnodata 0 -r bilinear DEM_MASKED_L93_20m_bilinear.tif DEM_MSF_SIN_375m_bilinear.tif
# and for lanczos
gdalwarp -t_srs "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs" -tr 370.650173222222 370.650173222222 -te -420000 4486309.549622223 877275.6062777771 5450000 -srcnodata 0 -r lanczos DEM_MASKED_L93_20m_bilinear.tif DEM_MSF_SIN_375m_lanczos.tif

# Then for slope
gdaldem slope -alg horn DEM_MSF_SIN_375m_bilinear.tif SLP_MSF_SIN_375m_bilinear.tif
gdaldem slope -alg horn DEM_MSF_SIN_375m_lanczos.tif SLP_MSF_SO?_375m_lanczos.tif

# Forest mask
gdalwarp -t_srs "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs" -tr 370.650173222222 370.650173222222 -te -420000 4486309.549622223 877275.6062777771 5450000 -r nearest europe/corine_2006_forest_mask_europe.tif corine_2006_forest_mask_sin_375m.tif
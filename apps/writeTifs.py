import os
from osgeo import gdal
import glob

# Set GDAL retry and delay environment variables
os.environ['GDAL_HTTP_MAX_RETRY'] = '10'  # Number of retries
os.environ['GDAL_HTTP_RETRY_DELAY'] = '10'

# List of VRT files
vrt_files = glob.glob("/home/tim/dentata/Sentinel2_seasonal/*.vrt")

# Input file for extent
input_file = "/home/tim/rubella/scripts/SpectralBotany/data/BBSaoi.tif"

# Get the extent of the input file
input_ds = gdal.Open(input_file)
input_extent = input_ds.GetGeoTransform()
input_proj = input_ds.GetProjection()

# Print the input extent for debugging
print("Input Extent (GeoTransform):", input_extent)

# Calculate the bounds from the geotransform
minX = input_extent[0]
maxY = input_extent[3]
maxX = minX + input_extent[1] * input_ds.RasterXSize
minY = maxY + input_extent[5] * input_ds.RasterYSize

output_bounds = [minX, minY, maxX, maxY]

# Print the calculated bounds for debugging
print("Calculated Output Bounds:", output_bounds)

# Define a progress callback function
def progress_callback(complete, message, unknown):
    print(f"\rProgress: {complete * 100:.2f}% - {message}", end='')

# Iterate through VRT files and use gdal.Warp to write TIFs with progress callback
for vrt in vrt_files:
    try:
        output_tif = vrt.replace(".vrt", "_brig.tif")
        gdal.Warp(output_tif, vrt, outputBounds=output_bounds, dstSRS=input_proj, callback=progress_callback)

    except Exception as e:
        print(vrt + " failed")
        print(e)

# Close the input dataset
input_ds = None
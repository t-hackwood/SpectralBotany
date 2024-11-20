# Imports
from rios import applier, cuiprogress, pixelgrid
import numpy as np
import os
from osgeo import gdal, ogr, gdalconst, osr
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import joblib
from rios import applier, cuiprogress
import glob
import re 
from numba import njit, jit


@njit
def numba_median(nbar, nodata, minObs=3):
    """Calculate the median of a stack of images, ignoring no data values."""
    
    xsize, ysize, nfiles = nbar.shape
    
    output = np.empty((1,xsize, ysize), dtype=nbar.dtype)
    
    for x in range(xsize):
        for y in range(ysize):
            # Get the values for this pixel
            values = nbar[x, y, :]
            # Count the number of observations
            nobs = np.sum(values != nodata)
            # If there are enough observations, calculate the median
            if nobs >= minObs:
                output[:,x,y] = np.median(values[values != nodata])
            else:
                output[:,x,y] = nodata
                            
    return output


@njit
def numba_stddev(nbar, nodata, minObs=3):
    """Calculate the standard deviation of a stack of images, ignoring no data values."""
    
    xsize, ysize, nfiles = nbar.shape
    
    output = np.empty((1,xsize, ysize), dtype=nbar.dtype)
    
    for x in range(xsize):
        for y in range(ysize):
            # Get the values for this pixel
            values = nbar[x, y, :]
            # Count the number of observations
            nobs = np.sum(values != nodata)
            # If there are enough observations, calculate the standard deviation
            if nobs >= minObs:
                output[:,x,y] = np.std(values[values != nodata])
            else:
                output[:,x,y] = nodata
            
    return output


def numba_percentiles(arr, nodata, minObs=3, percentiles=[5, 25, 75, 95]):
    xsize, ysize, nfiles = arr.shape
    npercentiles = len(percentiles)
    
    median_output = np.empty((1, xsize, ysize), dtype=arr.dtype)
    stddev_output = np.empty((1, xsize, ysize), dtype=arr.dtype)
    percentile_output = np.empty((npercentiles, xsize, ysize), dtype=arr.dtype)
    
    for x in range(xsize):
        for y in range(ysize):
            values = arr[x, y, :]
            nobs = np.sum(values != nodata)
            if nobs >= minObs:
                valid_values = values[values != nodata]
                median_output[:, x, y] = np.median(valid_values)
                stddev_output[:, x, y] = np.std(valid_values)
                percentile_output[:, x, y] = np.percentile(valid_values, percentiles)
            else:
                median_output[:, x, y] = nodata
                stddev_output[:, x, y] = nodata
                percentile_output[:, x, y] = np.full(npercentiles, nodata)
                
    return median_output, stddev_output, percentile_output



def _vegIndices(info, inputs, outputs, otherargs):
    """
    Get vegetation and water index percentiles/stddev/range as proxy for phenology.
    """
    # Open the images
    s2 = inputs.inlist
    aoi = inputs.aoi

    ndvistack = []
    ndwistack = []

    for img in s2:
        
        nodata = np.any(img == otherargs.noData, axis=0)

        inshape = img.shape
        ndvi = (img[3]-img[2])/(img[3]+img[2]+np.finfo(float).eps).reshape(1, inshape[1], inshape[2])
        ndwi = (img[1]-img[5])/(img[1]+img[5]+np.finfo(float).eps).reshape(1, inshape[1], inshape[2])
        
        # Convert to 16 bit
        ndvi = 10000 + 10000 * ndvi
        ndwi = 10000 + 10000 * ndwi
        
        # Clip to np.uint16 values
        ndvi = np.clip(ndvi, 0, 65534).astype(np.uint16)
        ndwi = np.clip(ndwi, 0, 65534).astype(np.uint16)
        
        ndvi[:,nodata] = 0
        ndvi[aoi == 0] = 0
        ndwi[:,nodata] = 0
        ndwi[aoi == 0] = 0
        
        ndvistack.append(ndvi)
        ndwistack.append(ndwi)       
            
    ndvistack = np.stack(ndvistack, axis=0)
    ndwistack = np.stack(ndwistack, axis=0)
    
    # Transpose to x, y, time
    ndvistack = np.transpose(np.squeeze(ndvistack), (1, 2, 0))
    ndwistack = np.transpose(np.squeeze(ndwistack), (1, 2, 0))
    
    ndvimedian, ndvistd, ndviperc = numba_percentiles(ndvistack, 0)
    ndwimedian, ndwistd, ndwiperc = numba_percentiles(ndwistack, 0)
    
    ndviStats = np.vstack((ndvimedian, ndvistd, ndviperc))
    ndwiStats = np.vstack((ndwimedian, ndwistd, ndwiperc))
    
    outputs.ndvi =  ndviStats.astype(np.uint16)
    outputs.ndwi =  ndwiStats.astype(np.uint16)

# Get the no data value
ds = gdal.Open("/home/tim/dentata/Sentinel2_seasonal/cvmsre_qld_m202109202111_abma2.vrt")
noData = ds.GetRasterBand(1).GetNoDataValue()
res = ds.GetGeoTransform()[1]

# Use glob to get all .vrt files
all_files = glob.glob("/home/tim/dentata/Sentinel2_seasonal/*_brig.tif")

# Filter the list to include only those filenames that contain the years 2019-2024
pattern = re.compile(r'20(19|20|21|22|23)')
filelist = [f for f in all_files if pattern.search(f)]

print(filelist)
    
# Create the RIOS file objects
infiles = applier.FilenameAssociations()
outfiles = applier.FilenameAssociations()

# Setup the IO
infiles.aoi = "/home/tim/rubella/scripts/SpectralBotany/data/BBSaoi.tif"
infiles.inlist = filelist

outfiles.ndvi = "/home/tim/dentata/outputs/brigalow_ndvistats.tif"
outfiles.ndwi = "/home/tim/dentata/outputs/brigalow_ndwistats.tif"

# Get the otherargs
otherargs = applier.OtherInputs()

otherargs.noData = noData

# Controls for the processing   
controls = applier.ApplierControls()
controls.windowxsize = 512
controls.windowysize = 512
controls.setStatsIgnore(0) #  nodata
controls.progress = cuiprogress.CUIProgressBar()
controls.setReferenceImage(referenceImage="aoi")
controls.setFootprintType("INTERSECTION")
controls.setResampleMethod("near")
controls.setOutputDriverName("GTIFF")
controls.setCreationOptions(["COMPRESS=DEFLATE",
                                "ZLEVEL=9",
                                "PREDICTOR=2",
                                "BIGTIFF=YES",
                                "TILED=YES",
                                "INTERLEAVE=BAND",
                                "NUM_THREADS=ALL_CPUS",
                                "BLOCKXSIZE=512",
                                "BLOCKYSIZE=512"])

# Set concurrency depending on system
conc = applier.ConcurrencyStyle(numReadWorkers=3,
                                numComputeWorkers=2,
                                computeWorkerKind="CW_THREADS",
                                readBufferPopTimeout=300,
                                computeBufferPopTimeout=300,
                                readBufferInsertTimeout=300
                                )

controls.setConcurrencyStyle(conc)

def main():
    # Run the function
    print("Processing veg indices")
    applier.apply(_vegIndices, infiles, outfiles, otherargs, controls=controls)

if __name__ == "__main__":
    main() 
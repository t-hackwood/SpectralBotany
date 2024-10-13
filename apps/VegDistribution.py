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
        
        ndvi[:,nodata] = np.nan
        ndwistack[:,nodata] = np.nan
            
        # Stack these ratios with the original data
        ndvistack.append(ndvi)
        ndwistack.append(ndwi)
            
    ndvistack = np.stack(ndvistack, axis=0)
    ndwistack = np.stack(ndwistack, axis=0)

    # Get new stack with 5th, 50th and 95th percentiles and standard deviation
    ndviStats = np.nanpercentile(ndvistack, [5, 50, 95], axis=0)
    ndviStats = np.squeeze(ndviStats)
    ndviSTD = np.nanstd(ndvistack, axis=0)
    ndvirange = np.max(ndvistack, axis=0) - np.min(ndvistack, axis=0)
    ndviStack = np.vstack([ndviStats, ndviSTD, ndvirange])

    ndwiStats = np.nanpercentile(ndwistack, [5, 50, 95], axis=0)
    ndwiStats = np.squeeze(ndwiStats)
    ndwiSTD = np.nanstd(ndwistack, axis=0)
    ndwirange = np.max(ndwistack, axis=0) - np.min(ndwistack, axis=0)
    ndwiStack = np.vstack([ndwiStats, ndwiSTD, ndwirange])
    
    # Convert nans back to no data
    ndviStack[np.isnan(ndviStack)] = 0

    # rescale to 16bit
    ndviStack = 10000 * 10000 + ndviStack
    ndwiStack = 10000 * 10000 + ndwiStack

    ndviStack[:,np.any(s2 == otherargs.noData,axis=0)] = 0
    ndviStack[:, np.any(aoi == 0, axis=0)] = 0 # mask to the AOI
    outputs.ndvi =  ndviStack.astype(np.uint16)

    ndwiStack[:,np.any(s2 == otherargs.noData,axis=0)] = 0
    ndwiStack[:, np.any(aoi == 255, axis=0)] = 0 # mask to the AOI
    outputs.ndwi =  ndwiStack.astype(np.uint16)

# Get the no data value
ds = gdal.Open("/home/tim/dentata/Sentinel2_seasonal/cvmsre_qld_m202109202111_abma2.vrt")
noData = ds.GetRasterBand(1).GetNoDataValue()
res = ds.GetGeoTransform()[1]

# Use glob to get all .vrt files
all_files = glob.glob("/home/tim/dentata/Sentinel2_seasonal/*.tif")

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
controls.setFootprintType("BOUNDS_FROM_REFERENCE")
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
                                readBufferPopTimeout=600,
                                computeBufferPopTimeout=600
                                )

controls.setConcurrencyStyle(conc)

def main():
    # Run the function
    print("Processing veg indices")
    applier.apply(_vegIndices, infiles, outfiles, otherargs, controls=controls)

if __name__ == "__main__":
    main() 
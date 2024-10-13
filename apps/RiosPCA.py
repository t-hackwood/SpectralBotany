# Imports
from rios import applier, cuiprogress, pixelgrid, fileinfo
import numpy as np
import os
from osgeo import gdal, ogr, gdalconst, osr
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import joblib
import glob



# FIles for PCA and Segment outputs
RASTER_PC = "/home/tim/rubella/scripts/SpectralBotany/data/Sentinel/Sentinel_brigalow_PCA.tif" 

# Apply the PCA to the full image and mask to Brigalow AOI
# Takes ~20 minutes
def _applyPCA(info, inputs, outputs, otherargs):
    """
    Apply PCA to full resolution dataset.
    """
    # Open the images
    img = inputs.raster
    inshape = img.shape
    aoi = inputs.aoi
    
    # Get indicies
    ndvi = (img[3]-img[2])/(img[3]+img[2]+np.finfo(float).eps).reshape(1, inshape[1], inshape[2])
    ndwi = (img[1]-img[5])/(img[1]+img[5]+np.finfo(float).eps).reshape(1, inshape[1], inshape[2])

        
    # Stack these ratios with the original data
    stack = np.vstack([img, ndvi, ndwi])
    scaled_stack = np.reshape(stack, (stack.shape[0], -1)).astype('float32').T

    # Apply the PCA    
    pc = otherargs.pca.transform(otherargs.scaler.transform(scaled_stack))
    # Rescale to 16bit
    pc = np.round(np.clip(1.0 + 65534.0
                        * (pc-otherargs.bytescale[0])
                        / (otherargs.bytescale[1]-otherargs.bytescale[0])
                        ,1,65535))
        
    # Reshape the output
    pc = np.reshape(pc.T,(pc.shape[1],inshape[1],inshape[2]))
    # Mask the output for no data

    pc[:,np.any(stack == otherargs.noData,axis=0)] = 0
    pc[:, np.any(aoi == 255, axis=0)] = 0 # mask to the AOI
    outputs.pc =  pc.astype(np.uint16)

# Get the no data value
ds = gdal.Open("/home/tim/dentata/Sentinel2_seasonal/cvmsre_qld_m202312202402_abma2.vrt")
noData = ds.GetRasterBand(1).GetNoDataValue()
    
# Create the RIOS file objects
infiles = applier.FilenameAssociations()
outfiles = applier.FilenameAssociations()

# Setup the IO
infiles.aoi = "/home/tim/rubella/scripts/SpectralBotany/data/BBSaoi.tif"
infiles.raster = "/home/tim/dentata/Sentinel2_seasonal/cvmsre_qld_m202312202402_abma2.vrt"

outfiles.pc = RASTER_PC

# Get the otherargs
otherargs = applier.OtherInputs()
otherargs.pca = joblib.load("/home/tim/rubella/scripts/SpectralBotany/pca.pkl")
otherargs.scaler = joblib.load("/home/tim/rubella/scripts/SpectralBotany/stack_scaler.pkl")
otherargs.bytescale = joblib.load("/home/tim/rubella/scripts/SpectralBotany/byteScale.pkl")
otherargs.noData = noData

# Controls for the processing   
controls = applier.ApplierControls()
controls.windowxsize = 512
controls.windowysize = 512
controls.setReferenceImage(referenceImage="aoi")
controls.setFootprintType("INTERSECTION")
controls.setResampleMethod("near")
controls.setStatsIgnore(0) #  nodata
controls.progress = cuiprogress.CUIProgressBar()
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
                                readBufferPopTimeout=360,
                                computeBufferPopTimeout=360
                                )

controls.setConcurrencyStyle(conc)


def main():
    # Run the function
    print("Processing PCA")
    rtn = applier.apply(_applyPCA, infiles, outfiles, otherargs, controls=controls)
    print(rtn.timings.formatReport())


if __name__ == "__main__":
    main() 
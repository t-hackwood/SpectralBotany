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
RASTER_PC = "/home/tim/rubella/scripts/SpectralBotany/data/Sentinel/Sentinel_brigalow_PCA_v2.tif" 

# Apply the PCA to the full image and mask to Brigalow AOI
# Takes ~20 minutes
# Apply the PCA to the full image and mask to Brigalow AOI
# Takes ~20 minutes

def all_ratios(sampleData):
    numBands = sampleData.shape[0]
    clippedData = np.clip(sampleData, 0, None).astype(float)
    # Get indices of upper triangle of the array
    i, j = np.triu_indices(numBands, 1)
    # Calculate the ratio of each band to every other band
    ratio_arr = (clippedData[i]-clippedData[j])/(clippedData[i]+clippedData[j]+np.finfo(float).eps)
    # Concatenate sampleData and reshaped ratio_arr
    sampleData = np.concatenate((sampleData,ratio_arr.reshape((-1,) + sampleData.shape[1:])), axis=0)
    
    return sampleData



def _applyPCA(info, inputs, outputs, otherargs):
    """
    Apply PCA to full resolution dataset.
    """
    # Open the images
    s2 = inputs.inlist
    inshape = s2[0].shape
    aoi = inputs.aoi
    # nodata = np.any(s2[0] == 0, axis=0)

    datastack = []

    for img in s2:

        img = all_ratios(img)
        
        inshape = img.shape        
        ndvi = (img[3]-img[2])/(img[3]+img[2]+np.finfo(float).eps).reshape(1, inshape[1], inshape[2])
        ndwi = (img[1]-img[5])/(img[1]+img[5]+np.finfo(float).eps).reshape(1, inshape[1], inshape[2])
            
        # Stack these ratios with the original data
        stack = np.vstack([img, ndvi, ndwi])
        
        datastack.append(stack)
            
    stack = np.vstack(datastack)

    # Calculate the mean and std across all images (axis 0) for each band
    # mean_bands = np.mean(datastack, axis=0)
    # std_bands = np.std(datastack, axis=0)
    # stdOut = 10000 * 10000 + std_bands
    # print(stdOut.shape)
    # stdOut[:, np.any(nodata)] = 0
    # outputs.ndviSD = stdOut.astype(np.uint16)

    # # Stack the mean and std arrays to form a new data stack
    # stack = np.concatenate([mean_bands, std_bands], axis=0)

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
    pc[:, np.any(aoi == 0, axis=0)] = 0 # mask to the AOI
    outputs.pc =  pc.astype(np.uint16)

# Get the no data value
ds = gdal.Open("/home/tim/dentata/Sentinel2_seasonal/cvmsre_qld_m202109202111_abma2.vrt")
noData = ds.GetRasterBand(1).GetNoDataValue()
    
# Create the RIOS file objects
infiles = applier.FilenameAssociations()
outfiles = applier.FilenameAssociations()

# Setup the IO
infiles.inlist = ["/home/tim/dentata/Sentinel2_seasonal/cvmsre_qld_m202309202311_abma2_brig.tif"]
infiles.aoi = "/home/tim/rubella/scripts/SpectralBotany/data/SpectralBotanyTest.gpkg"

outfiles.pc = RASTER_PC
# outfiles.ndviSD = "./NDVI_SD"

# Get the otherargs
otherargs = applier.OtherInputs()
otherargs.pca = joblib.load("/home/tim/rubella/scripts/SpectralBotany/pca.pkl")
otherargs.scaler = joblib.load("/home/tim/rubella/scripts/SpectralBotany/stack_scaler.pkl")
otherargs.bytescale = joblib.load("/home/tim/rubella/scripts/SpectralBotany/byteScale.pkl")
otherargs.noData = noData

# Controls for the processing   
controls = applier.ApplierControls()
controls.vectorlayer = "BrigalowBeltAOI"
controls.setBurnValue = 1
controls.windowxsize = 512
controls.windowysize = 512
controls.setStatsIgnore(0) #  nodata
controls.progress = cuiprogress.CUIProgressBar()
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
                                readBufferPopTimeout=120,
                                computeBufferPopTimeout=120
                                )

controls.setConcurrencyStyle(conc)


def main():
    # Run the function
    print("Processing PCA")
    rtn = applier.apply(_applyPCA, infiles, outfiles, otherargs, controls=controls)
    print(rtn.timings.formatReport())


if __name__ == "__main__":
    main() 
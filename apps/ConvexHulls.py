from rios import applier, cuiprogress, ratapplier, rat
from scipy.spatial import ConvexHull
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
import joblib

SEGIDs = "/home/tim/rubella/scripts/SpectralBotany/data/Sentinel/Sentinel_brigalow_PCA_e24_v2_segs_2000_id.tif"

PCA = "/home/tim/rubella/scripts/SpectralBotany/data/Sentinel/Sentinel_brigalow_PCA_e24_v2.tif"


# # FIles for PCA and Segment outputs
# RASTER_PC = "data/Sentinel/Sentinel_brigalow_PCA_e24.tif"
RASTER_SEG = '/home/tim/rubella/scripts/SpectralBotany/data/Sentinel/Sentinel_brigalow_PCA_e24_v2_segs_2000.kea'

def _getSegmentHull(info, inputs, outputs, otherargs):
    segments = inputs.segs[0]
    pca = inputs.PCA
    
    # Get unique segment IDs
    segIDs = np.unique(segments)
            
    # Get the PCA values for each segment
    for segID in segIDs:
        if segID > 0:
            mask = segments == segID
            pca_values = pca[:, mask]
            # Reshape for scaling
            pca_values = np.reshape(pca_values, (pca_values.shape[0], -1)).T
            pca_length = len(pca_values)
            
            if pca_length > 3:
                
                pca_values = otherargs.scaler.transform(pca_values)
            
                
                # Calculate the convex hull volume
                try:
                    hull = ConvexHull(pca_values)
                    hullVol = hull.volume
                    # print(f"Hull Volume for Segment {segID}: {hullVol}")
                    
                    # Check if there's already a value in the dictionary
                    existing_vol, existing_length = otherargs.hullVols.get(segID, (0, 0))
                    # print(f"Existing Volume: {existing_vol}, Existing Length: {existing_length}")
                    
                    # Compare lengths and update dictionary
                    if hullVol > existing_vol:
                        # print(f"Updating Segment {segID}: New Length {pca_length}, Old Length {existing_length}")
                        otherargs.hullVols[segID] = (hullVol, pca_length)
                    else:
                        otherargs.hullVols[segID] = (existing_vol, existing_length)
                except Exception as e:
                    print(f"Segment ID: {segID}, PCA Length: {pca_length}")
                    print(f"Error calculating hull for Segment {segID}: {e}")
                    # If there's an error, keep the existing value or set to 65535 if not present
                    existing_vol, existing_length = otherargs.hullVols.get(segID, (0, 0))
                    if existing_length == 0:
                        otherargs.hullVols[segID] = (65535, existing_length)
            

        else:
            existing_vol, existing_length = otherargs.hullVols.get(segID, (0, 0))
            otherargs.hullVols[segID] = (max(existing_vol, 0), 0)
            

def getSegmentHull(segidRaster, PCAraster):    
    # Create the RIOS file objects
    infiles = applier.FilenameAssociations()
    outfiles = applier.FilenameAssociations()

    # Setup the IO
    infiles.segs = segidRaster
    infiles.PCA = PCAraster

    # Get the otherargs
    otherargs = applier.OtherInputs()
    otherargs.hullVols = {}
    otherargs.scaler = joblib.load("/home/tim/rubella/scripts/SpectralBotany/pcascaler.pkl")

    # Controls for the processing   
    controls = applier.ApplierControls()
    controls.windowxsize = 2048
    controls.windowysize = 2048
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
    controls.setOverlap = 1024

    # # Set concurrency depending on system
    # conc = applier.ConcurrencyStyle(numReadWorkers=5,
    #                                 numComputeWorkers=4,
    #                                 computeWorkerKind="CW_THREADS",
    #                                 readBufferInsertTimeout=200,
    #                                 computeBufferInsertTimeout=200,
    #                                 readBufferPopTimeout=200,
    #                                 computeBufferPopTimeout=200,
    #                                 )

    # controls.setConcurrencyStyle(conc)

    # Run the function
    applier.apply(_getSegmentHull, infiles, outfiles, otherargs, controls=controls)
    
    return otherargs.hullVols
    
    
HullVols = getSegmentHull(SEGIDs, PCA)

print("saving hull volumes")
HullVols_str_keys = {str(key): value for key, value in HullVols.items()}
# Save Hull volumes to a json
with open("/home/tim/rubella/scripts/SpectralBotany/data/Sentinel/Sentinel_brigalow_PCA_e24_hull_vols_v5.json", "w") as f:
    json.dump(HullVols_str_keys, f)

lenhulls = len(HullVols)

# Get number of segments
hist = rat.readColumn(RASTER_SEG, 'Histogram')
print(len(hist))

# Load convex hull dictionary
with open('/home/tim/rubella/scripts/SpectralBotany/data/Sentinel/Sentinel_brigalow_PCA_e24_hull_vols_v5.json', 'r') as f:
    ConvexHulls = json.load(f)
    
print(len(ConvexHulls))

# Iterate through dictionary and covnert hull volumes to an array
hulls = []
for i in range(0, len(hist)):
    key = str(i)
    if key in ConvexHulls:
        hulls.append(ConvexHulls[key][0])
    else:
        hulls.append(65535)
        
hulls = np.array(hulls)
maxval = np.max(hulls[hulls != 65535])
minval = np.min(hulls[hulls != 65535])
# Set no data values to 65535
hulls[hulls == 0] = 65535

print(maxval, minval)

# Write the hulls to the segmentation
print('Writing Hulls to Segmentation')
rat.writeColumn(RASTER_SEG, 'convex_hulls', hulls)

print('Exporting Segmentation')
infiles = applier.FilenameAssociations()
outfiles = applier.FilenameAssociations()   
infiles.image = RASTER_SEG
outfiles.var = RASTER_SEG.replace('.kea', '_hulls_v5.tif')

otherargs = applier.OtherInputs()
# Using a loop to read and store each 0 column into otherargs
setattr(otherargs, 'hulls', 
            np.array(rat.readColumn(infiles.image, 'convex_hulls')).astype(np.uint16))

otherargs.noData = 65535
controls = applier.ApplierControls()
controls.windowxsize = controls.windowysize = 512  # Set both attributes on the same line
controls.setReferenceImage(RASTER_SEG)
controls.setStatsIgnore(65535)
controls.setOutputDriverName("GTIFF")
controls.setCreationOptions([
    "COMPRESS=DEFLATE",
    "ZLEVEL=9",
    "BIGTIFF=YES",
    "TILED=YES",
    "INTERLEAVE=BAND",
    "NUM_THREADS=ALL_CPUS",
    "BLOCKXSIZE=512",
    "BLOCKYSIZE=512"
])

def exportColor(info, inputs, outputs, otherargs):   
    data = inputs.image.flatten()
    # Access b1 through b3 from otherargs using a loop instead of individual lines
    rgb = getattr(otherargs, 'hulls')[data]
    outputs.var = rgb.reshape(inputs.image.shape).astype(np.uint16)

applier.apply(exportColor, infiles, outfiles, otherargs, controls=controls)
from bs4 import BeautifulSoup
import requests
from osgeo import gdal
import os

# GDAL Environment Variables
os.environ["GDAL_CACHEMAX"] = "1024000000"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"
os.environ["GDAL_HTTP_MULTIPLEX"] = "YES"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif,.TIF,.vrt"
os.environ["VSI_CACHE"] = "True"
os.environ["VSI_CACHE_SIZE"] = "1024000000"
os.environ["GDAL_HTTP_MAX_RETRY"] = "10"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "3"

# SR URL
url = 'https://data.tern.org.au/rs/public/data/sentinel2/surface_reflectance/qld/'

outDir = "/home/tim/dentata/Sentinel2_seasonal"

year = "2023"

# Create the output directory if it doesn't exist
if not os.path.exists(outDir):
    os.makedirs(outDir)

# Make a list of the queensland files on the TERN server
soup = BeautifulSoup(requests.get(url).text,"html.parser")
ternFiles = []
for a in soup.find_all('a'):
    link = a['href']
    if link[-4:] == '.tif':
        ternFiles.append(link)

worked = False
while worked == False:
    worked = True
    try:
        for ternFile in ternFiles:
            outName = os.path.join(outDir,ternFile).replace(".tif",".vrt")
            print('Processing:',ternFile)

            dslink = '/vsicurl/' + url + ternFile
            ds = gdal.BuildVRT(outName, dslink, resolution='highest', resampleAlg='cubic')
            ds = None
    except:
        worked = False


if __name__ == '__main__':
    print('Processing Complete')
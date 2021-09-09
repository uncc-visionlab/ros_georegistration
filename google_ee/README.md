# Google Maps and Earth Engine Collector
This script collects images of Google Maps sattelite images and images from the Google Earth Engine.

The datasets used from the Google Earth Engine are:
* Sentinel-1 (SAR)
* Sentinel-2 Surface Reflectance (Cloudy and Cloudless)
* USDA NAIP (EO)

An Earth Engine account is required to collect Earth Engine images and send them to your Google Drive.

There is also a quality of life improvement that allows the script to download and translate the collected images, however a Google Drive API App is required (discussed later).

If you choose to not make the Google Drive API App, then you will need to manually translate each image (described at the end).

## Map Downloader requirements for Python 2.7

Generic installs:

* Matplotlib

`pip install matplotlib`

* Salem

`pip install salem`

* Pandas

`pip install pandas`

* PyProj

`pip install pyproj`

* Xarray


`pip install xarray`

* Shapely

`pip install shapely`

* Descartes

`pip install descartes`

* ImageTk

`sudo apt-get install python-pil.imagetk`

* pyKML

`pip install pykml`

* Motionless

`pip install motionless`

* PyDrive

`pip install pydrive`

* SciPy

`pip install scipy`

* netCDF4 v1.5.5.1

`pip install netCDF4==1.5.5.1`

* joblib

`pip install joblib`

GDAL for Python install:
```
sudo add-apt-repository ppa:ubuntugis/ppa
sudo apt-get update && sudo apt-get upgrade
sudo apt install gdal-bin
sudo apt install libgdal-dev
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
pip install GDAL==<GDAL VERSION FROM OGRINFO> (ogrinfo --version)
```
Earth Engine install and one time authenticate script (need Earth Engine account):
```
pip install earthengine-api --upgrade
```
(inside of a python script)
```
import ee
ee.Authenticate()
```

## Google Drive App for PyDrive

In order to create a Google Drive API App, you must first go to the [Google API Console](https://console.cloud.google.com/) website and then:

1. Create a project.
2. Search for "Google Drive API" and enable it.
3. Go to the Credentials menu and create credentials with "OAuth client ID."
4. Enter an appropriate name and then click "Configure consent screen." Once finished:
    1. Set application type to "Web Application."
    2. Enter a name
    3. Set "Authorized JavaScript origins" to http://localhost:8080
    4. Set "Authorized redirect URIs" to http://localhost:8080/
    5. Save
5. Click "Download JSON" and rename it to "client_secrets.json" and place it in this directory

[(Original tutorial)](https://pythonhosted.org/PyDrive/quickstart.html#authentication)

## ee_mapDownloader.py

* Inputs
	* Argument 1:
		* manual
		* random
		* randomMulti
	
	* If argument 1 is manual:
		* The lat long coordinates being the next two arguments.
		* The number of pixels in x and y for the next two arguments.
		* The meter per pixel ratio for the final arguement.
		* sar, naip, cloudy, and cloudless can be used at the end as optional arguments.
		* EXAMPLE (w/o SAR or NAIP): 
			
			`python ee_mapDownloader.py manual 48.852411 -121.706750 5000 5000 1`
		
		* EXAMPLE (w/ SAR and NAIP):
		
			`python ee_mapDownloader.py manual 48.852411 -121.706750 5000 5000 1 sar naip`
		
		* GENERIC:
		
			`python ee_mapDownloader.py manual latCoord longCoord xPixels yPixels mpr [sar] [naip] [cloudy] [cloudless]`
			
	* If argument 1 is random:
		* The upper left corner lat long coordinates being the next 2 arguments.
		* The bottom right corner lat long coordinates for the next 2 arguments.
		* The number of pixels in x and y for the next two arguments.
		* The meter per pixel ratio for the final argument.
		* sar, naip, cloudy, and cloudless can be used at the end as optional arguements.
		* EXAMPLE (w/o SAR or NAIP):
			
			`python ee_mapDownloader.py random 48.852411 -121.706750 32.534790 -80.935663 5000 5000 1`
			
		* EXAMPLE (w/ SAR and NAIP):
		
			`python ee_mapDownloader.py random 48.852411 -121.706750 32.534790 -80.935663 5000 5000 5 sar naip`
			
		* GENERIC:
		
			`python ee_mapDownloader.py random latCoord longCoord latCoord longCoord xPixels yPixels mpr [sar] [naip] [cloudy] [cloudless]`
			
	* If argument 1 is randomMulti:
		* Same as if argument 1 is random, however the number of images must be after the meter per pixel argument.
		* EXAMPLE (w/o SAR or NAIP):
		
			`python ee_mapDownloader.py randomMulti 48.852411 -121.706750 32.534790 -80.935663 5000 5000 5`
		* EXAMPLE (w/ SAR or NAIP):
		
			`python ee_mapDownloader.py randomMulti 48.852411 -121.706750 32.534790 -80.935663 5000 5000 5 sar naip`
		* GENERIC:
		
			`python ee_mapDownloader.py randomMulti latCoord longCoord latCoord longCoord xPixels yPixels mpr numImages [sar] [naip] [cloudy] [cloudless]`
			
* Note: The latitude and longitude coordinate can be obtained by placing a pin in Google Maps.

## For manually translating Earth Engine images

Find the absolute Max/Min values over all the bands:

`gladinfo -mm image.tif`

For SAR (Sen-1) images:

`gdal_translate -of PNG -scale [MIN] [MAX] image.tif image.png`

For Sen-2 images: (Cloudless and Cloudy images)

`gdal_translate -of PNG -scale 0 [MAX] image.tif image.png`

For NAIP images:

`gdal_translate -of PNG image.tif image.png`

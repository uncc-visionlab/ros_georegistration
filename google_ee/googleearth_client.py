#!/usr/bin/env python3
# Install Google EarthEngine for python
# pip install earthengine-api --upgrade
# Install matplotlib
# sudo apt-get install python-matplotlib
# Install salem
# pip install salem
# Install pandas
# sudo apt-get install python-pandas
# Install pyproj
# sudo apt-get install python-pyproj
# Install netCDF4
# pip install netCDF4
# Install xarray
# pip install xarray
# Install shapely
# pip install shapely
# Install descartes
# pip install descartes
# Install imagetk
# sudo apt-get install python-pil.imagetk
# Install pykml
# pip install pykml
# Install geopandas
# sudo apt-get install python-geopandas
# Install motionless
# pip install motionless
import csv
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import salem
from salem import get_demo_file, DataLevels, GoogleVisibleMap, Map
from salem import GoogleCenterMap, gis, wgs84, utils

import time
import sys

import ee
import ee.geometry
import ee.mapclient
from ee.mapclient import MapClient, MapOverlay

import requests

from pykml import parser
from os import path
import os

import pyproj

from osgeo import gdal
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import datetime
import math

from osgeo import gdal

API_KEY = None

class MapTiles:
    def __init__(self, rows, cols):
        self.tiles = [[np.array([1,1],dtype=float) for j in range(cols)] for i in range(rows)]

class GoogleVisibleMap2(GoogleCenterMap):
    """Google static map automatically sized and zoomed to a selected region.

    It's usually more practical to use than GoogleCenterMap.
    """

    def __init__(self, x, y, crs=wgs84, size_x=640, size_y=640, scale=1,
                 maptype='satellite', use_cache=True, **kwargs):
        """Initialize

        Parameters
        ----------
        x : array
          x coordinates of the points to include on the map
        y : array
          y coordinates of the points to include on the map
        crs : proj or Grid
          coordinate reference system of x, y
        size_x : int
          image size
        size_y : int
          image size
        scale : int
          image scaling factor. 1, 2. 2 is higher resolution but takes
          longer to download
        maptype : str, default: 'satellite'
          'roadmap', 'satellite', 'hybrid', 'terrain'
        use_cache : bool, default: True
          store the downloaded image in the cache to avoid future downloads
        kwargs : **
          any keyword accepted by motionless.CenterMap (e.g. `key` for the API)

        Notes
        -----
        To obtain the exact domain specified in `x` and `y` you may have to
        play with the `size_x` and `size_y` kwargs.
        """

        global API_KEY

        if 'zoom' in kwargs or 'center_ll' in kwargs:
            raise ValueError('incompatible kwargs.')

        # Transform to lonlat
        crs = gis.check_crs(crs)
        if isinstance(crs, pyproj.Proj):
            lon, lat = gis.transform_proj(crs, wgs84, x, y)
        elif isinstance(crs, Grid):
            lon, lat = crs.ij_to_crs(x, y, crs=wgs84)
        else:
            raise NotImplementedError()

        # surely not the smartest way to do but should be enough for now
        mc = (np.mean(lon), np.mean(lat))
        zoom = 20
        while zoom >= 0:
            grid = gis.googlestatic_mercator_grid(center_ll=mc, nx=size_x,
                                                  ny=size_y, zoom=zoom,
                                                  scale=scale)
            dx, dy = grid.transform(lon, lat, maskout=True)
            if np.any(dx.mask):
                zoom -= 1
            else:
                break

        if 'key' not in kwargs:
            if API_KEY is None:
                with open(utils.get_demo_file('.api_key'), 'r') as f:
                    API_KEY = f.read().replace('\n', '')
            kwargs['key'] = API_KEY

        GoogleCenterMap.__init__(self, center_ll=mc, size_x=size_x,
                                 size_y=size_y, zoom=zoom, scale=scale,
                                 maptype=maptype, use_cache=use_cache, **kwargs)

def googlestatic_mercator_grid2(center_ll=None, nx=640, ny=640, zoom=12, scale=1):
    """Mercator map centered on a specified point (google API conventions).

    Mostly useful for google maps.
    """

    # Number of pixels in an image with a zoom level of 0.
    google_pix = 256 * scale
    # The equitorial radius of the Earth assuming WGS-84 ellipsoid.
    google_earth_radius = 6378137.0

    # Make a local proj
    lon, lat = center_ll
    #proj_params = dict(proj='merc', datum='WGS84')
    #projloc = pyproj.Proj(proj_params)
    projloc = pyproj.Proj("+init=EPSG:3857") 

    # The size of the image is multiplied by the scaling factor
    nx *= scale
    ny *= scale

    # Meter per pixel
    mpix = (2 * np.pi * google_earth_radius) / google_pix / (2**zoom)
    #mpix = (2 * np.pi * google_earth_radius) * np.cos(lat*np.pi/180.0) / google_pix / (2**zoom)
    xx = nx * mpix
    yy = ny * mpix

    e, n = pyproj.transform(wgs84, projloc, lon, lat)
    corner = (-xx / 2. + e, yy / 2. + n)
    dxdy = (xx / nx, - yy / ny)

    return gis.Grid(proj=projloc, x0y0=corner,
                nxny=(nx, ny), dxdy=dxdy,
                pixel_ref='corner')


class GoogleEarthClient():

    def __init__(self,args):
#        urlPath = image.getDownloadUrl({
#                                    'scale': 30,
#                                    'crs': 'EPSG:4326',
#                                    'region': '[[-120, 35], [-119, 35], [-119, 34], [-120, 34]]'
#                                    })
#        print ("%s" %str(urlPath))
#        print ("current path %s" %path.dirname(path.abspath(__file__)))
        
        #kml_file = path.join('./', 'J1_35580_STD_F373.kml')

        #with open(kml_file) as f:
        #    doc = parser.parse(f)
        #    print("doc %s" %str(doc))
                
        #shp = salem.read_shapefile('J1_35580_STD_F373-SHP/J1_35580_STD_F373-polygon.shp')
        timePrefix = str(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
        ufreef_input_params = { # UF REEF
            'latitude': 30.474693,   # degrees N
            'longitude': -86.572972, # degrees E
            'modelname': 'UF_REEF_SATELLITE',
            'rows' : 15,
            'columns': 15
        }
        willis_input_params = { # DR. WILLIS' HOME
            'latitude': 35.395703,   # degrees N
            'longitude': -80.535865, # degrees E
            'modelname': 'CONCORD_SATELLITE',
            'rows' : 15,
            'columns': 15
        }
        uncc_epic_input_params = { # UNCC EPIC
            'latitude': 35.309003,   # degrees N
            'longitude': -80.741692, # degrees E
            'modelname': 'UNCC_EPIC_SATELLITE',
            'rows' : 10,
            'columns': 10
        }
        eglin_lidar_range_input_params = { # EGLIN LIDAR RANGE
            'latitude': 30.563300,   # degrees N
            'longitude': -86.436442, # degrees E
            'modelname': 'EGLIN_LIDAR_RANGE_SATELLITE',
            'rows' : 18,
            'columns': 18
        }        
        afrl_rwwi_input_params = { # AFRL RWWI BUILDINGS
            'latitude': 30.480427,   # degrees N
            'longitude': -86.501005, # degrees E
            'modelname': 'AFRL_RWWI_SATELLITE',
            'rows' : 17,
            'columns': 17
        }
        kremmling_co_input_params = { # KREMMLING, CO AMMONITE FOSSIL PRESERVE
            'latitude': 40.230424,   # degrees N
            'longitude': -106.398358, # degrees E
            'modelname': 'KREMMLING_CO_SATELLITE',
            'rows' : 17,
            'columns': 17
        }
        estes_park_co_input_params = { # ROCKY MOUNTAIN NATIONAL PARK, ESTES PARK, CO
            'latitude': 40.413880,    # degrees N
            'longitude': -105.656720, # degrees E
            'modelname': 'ESTES_PARK_CO_SATELLITE',
            'rows' : 17,
            'columns': 17
        }
        albuquerque_input_params = { # Albuquerque
            'latitude' : 35.317139,  # degrees N
            'longitude': -106.844806,# degrees E
            'modelname': 'ALBUQUERQUE_SATELLITE',
            'rows' : 17,
            'columns': 17
        }
        isleta_pueblo_input_params = { #Isleta Pueblo, near Albuquerque
            'latitude' : 34.848778,   # degrees N
            'longitude': -106.492639,  # degrees E
            'modelname': 'ISLETA_PUEBLO_SATELLITE',
            'rows' : 17,
            'columns' : 17
        }
        if (args[1] == 'manual'):
            satList = args[7:]
            xLocation = float(args[3])
            yLocation = float(args[2])
            print("Latitude Coordinate: {} \nLongitude Coordinate: {}".format(yLocation,xLocation))
            manual = {
                'latitude' : yLocation, # degrees N
                'longitude': xLocation, # degrees E
                'modelname': 'RANDOM_SATELLITE',
                'rows' : int(args[5]),
                'columns' : int(args[4])
            }
            zoomChange = int(args[6])
            pathPrefix = "Batch_Manual_"+timePrefix+"/"
            os.mkdir(pathPrefix)
            fileBatch = self.generateMaps(manual,satList,1,pathPrefix,zoomChange)
            if (satList):
                self.driveDownloader(fileBatch)
        elif (args[1] == 'random'):
            import random
            satList = args[9:]
            yRange = abs(float(args[2])-float(args[4]))
            xRange = abs(float(args[3])-float(args[5]))
            yLocation = float(args[2])-random.random()*yRange
            xLocation = float(args[3])+random.random()*xRange
            print("Latitude Coordinate: {} \nLongitude Coordinate: {}".format(yLocation,xLocation))
            random = {
                'latitude' : yLocation, # degrees N
                'longitude': xLocation, # degrees E
                'modelname': 'RANDOM_SATELLITE',
                'rows' : int(args[7]),
                'columns' : int(args[6])
            }
            zoomChange = int(args[8])
            pathPrefix = "Batch_Random_"+timePrefix+"/"
            os.mkdir(pathPrefix)
            fileBatch = self.generateMaps(random,satList,1,pathPrefix,zoomChange)
            if (satList):
                self.driveDownloader(fileBatch)
        elif (args[1] == 'randomMulti'):
            import random
            os.mkdir("Batch_Multi_"+timePrefix)
            satList = args[10:]
            fileBatch = []
            zoomChange = int(args[8])
            for idx in range(0,int(args[9])):
                yRange = abs(float(args[2])-float(args[4]))
                xRange = abs(float(args[3])-float(args[5]))
                yLocation = float(args[2])-random.random()*yRange
                xLocation = float(args[3])+random.random()*xRange
                print("Latitude Coordinate: {} \nLongitude Coordinate: {}".format(yLocation,xLocation))
                randomMulti = {
                    'latitude' : yLocation, # degrees N
                    'longitude': xLocation, # degrees E
                    'modelname': 'RANDOM_SATELLITE',
                    'rows' : int(args[7]),
                    'columns' : int(args[6])
                }
                pathPrefix = "Batch_Multi_"+timePrefix+"/"+str(idx+1)+"/"
                os.mkdir(pathPrefix)
                temp = self.generateMaps(randomMulti,satList,idx,pathPrefix,zoomChange)
                for file in temp:
                    fileBatch.append(file)
            if (satList):
                self.driveDownloader(fileBatch)
        # CHANGE LOCATIONS HERE
        #input_params = uncc_epic_input_params
        #input_params = willis_input_params
        #input_params = albuquerque_input_params
        #input_params = eglin_lidar_range_input_params
        #input_params = kremmling_co_input_params
        #input_params = estes_park_co_input_params
        #input_params = afrl_rwwi_input_params
        #input_params = ufreef_input_params
        #input_params = isleta_pueblo_input_params
        #input_params = random
        
    def generateMaps(self,input_params,satList,setNumber,pathPrefix="",zoomChange=1):
        drive = None
        fileBatch = []
        CONSTRUCT_GAZEBO_PLANE_MODEL = True
        if (not satList):
            CONSTRUCT_SAR_REFERENCE_IMAGE = False
        else:
            CONSTRUCT_SAR_REFERENCE_IMAGE = True
        target_lon_lat = np.array([ input_params['longitude'],  input_params['latitude']])
        
        if (CONSTRUCT_GAZEBO_PLANE_MODEL == True):
            rows = input_params['rows']
            columns = input_params['columns']
            #rows = 17
            #columns = 17
            imageParams = self.downloadGoogleSatellite(target_lon_lat[0], target_lon_lat[1], rows, columns,pathPrefix,zoomChange)
            self.writeGazeboModelAsTarfile(input_params['modelname'], imageParams,pathPrefix)

        if (CONSTRUCT_SAR_REFERENCE_IMAGE == True and 'imageParams' in locals()):
            #ee.Authenticate()
            # read the shapefile and use its extent to define a ideally sized map
            ee.Initialize()
            #gauth = GoogleAuth(settings_file='DRIVE_API/settings.yaml')
            #image = ee.Image('srtm90_v4')
            #print(image.getInfo())
            #uf_reef_geom = ee.Geometry.Point(target_lat_lon.tolist());
            #self.test_EEBatchToDrive2()
            print('imageParams',imageParams)
            #target_geom = ee.Geometry.Point(target_lon_lat.tolist()).buffer(1000)
            eeRect_ptList = (imageParams['rect'][0], imageParams['rect'][2], imageParams['rect'][1], imageParams['rect'][3])
            target_geom = ee.Geometry.Rectangle(eeRect_ptList)
            #self.test_EEBatchToDrive(target_lat_lon_point_geom)
            for sat in satList:
                filenameStr = self.test_EEBatchToDrive(target_geom, imageParams, sat.lower(),drive,pathPrefix)
                fileBatch.append([pathPrefix,filenameStr,sat])
        return fileBatch
    
        
    def downloadGoogleSatellite(self, center_longitude, center_latitude, rowsP = 2, columnsP = 2,pathPrefix="",zoomChange=1):
        # If you need to do a lot of maps you might want
        # to use an API key and set it here with key='YOUR_API_KEY'
        #min_pt = uf_reef_geom.geometries().get(0)
        #plt.axis([-50,50,0,10000])
        target_lon_lat = [ center_longitude, center_latitude]
        plt.ion();
        #plt.show();
        #fig, ax = plt.subplots()
        #f, ax1 = plt.subplots(figsize=(12, 5))
        lat_lon_offset = np.zeros([2],dtype=float);
        imgHeight = 615.0
        rows = int(math.ceil(rowsP/imgHeight))
        verticalOverlap = 25; # Google watermark is on the last 24 pixel rows
        imgHeight = 640
        imgWidth = 640.0
        columns = int(math.ceil(columnsP/imgWidth))
        horizontalOverlap = 0; # Google watermark is on the last 24 pixel rows
        imgWidth = 640
        mapTiles = MapTiles(rows, columns);
        themaptype = 'satellite'
        scaleValue = 1
        zoomValue = min(17-int(math.log(zoomChange,2)),20)
        numPixelsX = (imgWidth - horizontalOverlap) * columns;
        numPixelsY = (imgHeight - verticalOverlap) * rows;
        grid = googlestatic_mercator_grid2(center_ll=target_lon_lat, nx=numPixelsX,
                                                  ny=numPixelsY, zoom=zoomValue,
                                                  scale=scaleValue)
        epsg3857_rect_m_east_north = grid.extent
        min_m_east_north =  np.array([epsg3857_rect_m_east_north[0], epsg3857_rect_m_east_north[2]])
        max_m_east_north =  np.array([epsg3857_rect_m_east_north[1], epsg3857_rect_m_east_north[3]])
        range_m = max_m_east_north - min_m_east_north
        m_per_ypixel = np.abs(range_m[1]) / numPixelsY
        m_per_xpixel = np.abs(range_m[0]) / numPixelsX
        print ('EPSG:3857 rect = %s' %(epsg3857_rect_m_east_north))
        print ('Guess at (x,y) resolution (%f,%f) m/pixel' %(m_per_xpixel,m_per_ypixel))        
        epsg4326_rect_lon_lat = grid.extent_in_crs()
        min_lon_lat =  np.array([epsg4326_rect_lon_lat[0], epsg4326_rect_lon_lat[2]])
        max_lon_lat =  np.array([epsg4326_rect_lon_lat[1], epsg4326_rect_lon_lat[3]])
        #proj_epsg3857 = pyproj.Proj("+init=EPSG:3857")
        #proj_epsg4326 = pyproj.Proj("+init=EPSG:4326")
        #e, n = pyproj.transform(proj_epsg3857, proj_epsg4326, epsg3857_rect_m_east_north[0], epsg3857_rect_m_east_north[2])
        #min_lon_lat =  np.array([e, n])
        #e, n = pyproj.transform(proj_epsg3857, proj_epsg4326, epsg3857_rect_m_east_north[1], epsg3857_rect_m_east_north[3])
        #max_lon_lat =  np.array([e, n])
        #epsg4326a_rect_lon_lat = (min_lon_lat[0], max_lon_lat[0], min_lon_lat[1], max_lon_lat[1])
        #print ('EPSG:4326 rect = %s' % str(epsg4326a_rect_lon_lat))
        range_lon_lat = max_lon_lat - min_lon_lat        

        tile_center_coords = np.empty((rows,columns,2))
        for tileYIdx in range(0,rows):
            for tileXIdx in range(0,columns):
                tile_center_coords[tileYIdx][tileXIdx][0] = min_lon_lat[1] + range_lon_lat[1]*(tileYIdx+0.5)/rows
                tile_center_coords[tileYIdx][tileXIdx][1] = min_lon_lat[0] + range_lon_lat[0]*(tileXIdx+0.5)/columns
                #print('tile(%d,%d) has (lat,lon) center (%f,%f)' % (tileYIdx,tileXIdx,
                #               tile_center_coords[tileYIdx][tileXIdx][1],
                #               tile_center_coords[tileYIdx][tileXIdx][0]))
        # get closest integer pixel coordinate
        #dy, dx = grid.transform(center_longitude, center_latitude, nearest=True)        
        #from pyproj import Proj, transform
        # longitude first, latitude second.
        #m_east, m_north = transform(Proj(init='epsg:4326'), Proj(init='epsg:3857'), center_longitude, center_latitude)
        # output (meters east of 0, meters north of 0)
        #print('(%f, %f)' % (m_east, m_north))  
        # output (meters east of 0, meters north of 0): (-14314.651244750548, 6711665.883938471)
        #print(transform(Proj(init='epsg:3857'), Proj(init='epsg:4326'), m_east, m_north))  # longitude first, latitude second.
        #lon_rounded_to_intpixel, lat_rounded_to_intpixel = grid.ij_to_crs(dx, dy, crs=wgs84, nearest=False)
        #center_longitude = lon_rounded_to_intpixel
        #center_latitude = lat_rounded_to_intpixel

        for tileYIdx in range(0,rows):
            for tileXIdx in range(0,columns):
                print("Grabbing (X,Y) tile (%d,%d) of (%d,%d)" % (tileXIdx+1,tileYIdx+1,columns,rows))
                lat_lon_offset = [tile_center_coords[tileYIdx][tileXIdx][1],
                                  tile_center_coords[tileYIdx][tileXIdx][0]]
                if (False):
                    min_pt = [x - 0.0001 for x in lat_lon_offset];
                    max_pt = [x + 0.0001 for x in lat_lon_offset];
                    g = GoogleVisibleMap2(x=[min_pt[0], max_pt[0]], y=[min_pt[1], max_pt[1]],                        
                            size_x=imgWidth, size_y=imgHeight, scale=scaleValue,  # scale is for more details
                            crs=wgs84, maptype=themaptype)  # try out also: 'terrain'
                else:
                    g = GoogleCenterMap(center_ll=lat_lon_offset, size_x=imgWidth,
                            size_y=imgHeight, zoom=zoomValue, scale=scaleValue,
                            maptype=themaptype, use_cache=True)
                    imageGrid = googlestatic_mercator_grid2(center_ll=lat_lon_offset, nx=imgWidth, ny=imgHeight,
                                                    zoom=zoomValue, scale=scaleValue)
                    delta_lon_lat = imageGrid.extent_in_crs()
                    #print('min(Lat,Lon), max(Lat,Lon) = (%f,%f),(%f,%f)' % (delta_lon_lat[0], delta_lon_lat[2], delta_lon_lat[1], delta_lon_lat[3]))
                mapTiles.tiles[tileYIdx][tileXIdx] = np.array(g.get_vardata());
                ggl_img = g.get_vardata() 
                #sm = Map(g.grid, factor=1, countries=False)
                #sm = Map(g.grid, factor=1, countries=False)
                #im = ax1.imshow(ggl_img)
                #gg_imgplt.set_title('Google static map')
                #plt.tight_layout()
                #plt.draw()
                #plt.show()
                #plt.pause(3);
                #sm.set_scale_bar(location=(0.88, 0.94))  # add scale
                #plt.imsave('tile_x%d_y%d.png' %(tileXIdx,tileYIdx), ggl_img)
                
        mergedTile = None
        for tileYIdx in range(0,rows):
            tile = mapTiles.tiles[tileYIdx][0]
            mergedRow = tile
            for tileXIdx in range(1,columns):                
                #mergedRow[imgHeight-verticalOverlap:imgHeight:imgHeight] = tile[0:verticalOverlap][:][:];
                tile = mapTiles.tiles[tileYIdx][tileXIdx]
                mergedRow = np.concatenate((mergedRow, tile), axis = 1)
                #mapTiles.tiles[tileYIdx][tileXIdx] = np.empty([1,1]);
            if tileYIdx == 0:
                clippedTile = mergedRow[0:imgHeight-verticalOverlap][:][:]
                mergedTile = clippedTile
                #mergedTile = mergedRow
            else:
                mergedRow[imgHeight-verticalOverlap:imgHeight][:][:] = 0
                clippedTile = mergedRow[0:imgHeight-verticalOverlap][:][:]
                mergedTile = np.concatenate((clippedTile, mergedTile), axis = 0)
                #mapTiles.tiles[tileYIdx][0] = np.empty([1,1]);

        # free memory
        for tileYIdx in range(rows-1,0,-1):
            for tileXIdx in range(columns-1,0,-1):
                del mapTiles.tiles[tileYIdx][tileXIdx]
        del mapTiles

        grid = googlestatic_mercator_grid2(center_ll=target_lon_lat, nx=columnsP,
                                                  ny=rowsP, zoom=zoomValue,
                                                  scale=scaleValue)
        epsg3857_rect_m_east_north = grid.extent
        min_m_east_north =  np.array([epsg3857_rect_m_east_north[0], epsg3857_rect_m_east_north[2]])
        max_m_east_north =  np.array([epsg3857_rect_m_east_north[1], epsg3857_rect_m_east_north[3]])
        range_m = max_m_east_north - min_m_east_north
        m_per_ypixel = np.abs(range_m[1]) / rowsP
        m_per_xpixel = np.abs(range_m[0]) / columnsP        
        epsg4326_rect_lon_lat = grid.extent_in_crs()
        min_lon_lat =  np.array([epsg4326_rect_lon_lat[0], epsg4326_rect_lon_lat[2]])
        max_lon_lat =  np.array([epsg4326_rect_lon_lat[1], epsg4326_rect_lon_lat[3]])

        diffX = int(columnsP/2)
        diffY = int(rowsP/2)
        centerX = int(numPixelsX/2)
        centerY = int(numPixelsY/2)
        
        print(mergedTile.shape)
        print("Range: {} - {}, {} - {}".format(centerX-diffX,centerX+diffX,centerY-diffY,centerY+diffY))
        mergedTile = mergedTile[centerY-diffY:centerY+diffY,centerX-diffX:centerX+diffX,:]

        [outputHeight, outputWidth, comps] = mergedTile.shape;
        width_m = outputWidth*m_per_xpixel
        height_m = outputHeight*m_per_ypixel
        img_filename = 'satellite_%sE_%sN_%dx_%dy_%dm_EW_%dm_NS.png' % (center_longitude, center_latitude, \
                    outputWidth, outputHeight, width_m, height_m)
        print('Creating image file %s' % str(pathPrefix+img_filename))
        plt.imsave(pathPrefix+img_filename, mergedTile)
        image_params = {
                'filename': img_filename,
                'xy_dimensions_m': [width_m, height_m],
                'xy_dimensions_pixels' : [outputWidth, outputHeight],
                'xy_pixels_per_m' : [m_per_xpixel, m_per_ypixel],
                'min_lon_lat' : min_lon_lat,
                'max_lon_lat' : max_lon_lat,
                #'rect' : epsg3857_rect_m_east_north,
                #'crs' : 'EPSG:3857',
                'rect' : epsg4326_rect_lon_lat,
                'crs' : 'EPSG:4326',
                'latitude': center_latitude,
                'longitude': center_longitude
                }
        return image_params
    
    def writeGazeboModelAsTarfile(self, model_name, image_params,pathPrefix=""):
        template_params = {
            'xy_dimensions_m':['%SIZE_METERS_X%','%SIZE_METERS_Y%'],
            'filename': '%TEXTURE_IMAGE_NAME%',
            'model_name' : '%MODEL_NAME%'
        }

        image_params['model_name'] = model_name

        strSDF = self.replaceTemplateParameters('satellite_ground_plane_template/model.sdf', template_params, image_params)
        strConfig = self.replaceTemplateParameters('satellite_ground_plane_template/model.config', template_params, image_params)
        strMaterial = self.replaceTemplateParameters('satellite_ground_plane_template/ground_plane_satellite.material', template_params, image_params)
        tarfilename = pathPrefix + model_name + ".tar.gz"
        
        import tarfile

        print('Creating gazebo model tar file %s' % str(tarfilename))
        tarfile_obj = tarfile.TarFile(tarfilename,"w")
        self.writeStringToTARFile(tarfile_obj, "model.sdf", strSDF)
        self.writeStringToTARFile(tarfile_obj, "model.config", strConfig)
        self.writeStringToTARFile(tarfile_obj, model_name + ".material", strMaterial)
        tarfile_obj.add(pathPrefix+image_params['filename'])
        tarfile_obj.close()
        
    def replaceTemplateParameters(self, filename, template_params, image_params):
        #input file
        fin = open(filename, "rt")
        fileContents = '';
        for line in fin:
            #read replace the string and write to output file
            line = line.replace(template_params['xy_dimensions_m'][0], str(image_params['xy_dimensions_m'][0]))
            line = line.replace(template_params['xy_dimensions_m'][1], str(image_params['xy_dimensions_m'][1]))
            line = line.replace(template_params['filename'], str(image_params['filename']))
            line = line.replace(template_params['model_name'], str(image_params['model_name']))
            fileContents = fileContents + line
        #print(fileContents)
        return fileContents
        
    def writeStringToTARFile(self, tarfile_obj, destination_filename, string_value):
        import tarfile
        import StringIO

        string_obj = StringIO.StringIO()
        string_obj.write(string_value)
        string_obj.seek(0)
        info = tarfile.TarInfo(name = destination_filename)
        info.size=len(string_obj.buf)
        tarfile_obj.addfile(tarinfo=info, fileobj=string_obj)
        
    def test_EEBatchToDrive2(self):
        #landsat = ee.ImageCollection("LANDSAT/LC08/C01/T1").filterDate('2016-01-01', '2017-01-01').filterBounds(uf_reef_geom)
        uf_reef_lat_lon = np.array([ -86.572972, 30.474693])
        uf_reef_geom = ee.Geometry.Point(uf_reef_lat_lon.tolist())
        #uf_reef_geom2 = ee.Geometry.LineString([[-120, 35], [-119, 35], [-119, 34], [-120, 34]])
        landsat = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR') \
                    .select(['B4', 'B3', 'B2']) \
                    .filterBounds(uf_reef_geom) \
                    .sort('CLOUD_COVER', False)  
        listOfImages = landsat.toList(10);
        landsat_image = ee.Image(listOfImages.get(0))
        urlpath = landsat_image.getDownloadUrl({
                    'scale': 120,
                    'crs': 'EPSG:4326',
                    'region': '[[-120, 35], [-119, 35], [-119, 34], [-120, 34]]'
                    })
        #print(urlpath)
        
        r = requests.get(urlpath, allow_redirects=True)
        print(r.headers.get('content-type'))
        open('download.zip', 'wb').write(r.content)
        #plt.imshow(r)
        # Google Satellite
        url = 'https://mt1.google.com/vt/lyrs=s&x=%d&y=%d&z=%d'
        #opt_overlay = ee.mapclient.MakeOverlay(url)        
        ee.mapclient.centerMap(-86.572972, 30.474693, 20)            
        #ee.mapclient.addToMap(imgCollection, vizParams, sat)      

    def function22(image, geom):
        return image.clip(geom)
        
    def maskS2clouds(self, image):
        qa = image.select('QA60')
        
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11
        
        img1 = qa.bitwiseAnd(cloudBitMask).eq(0)
        img2 = qa.bitwiseAnd(cirrusBitMask).eq(0)
        mask = img1 and img2
        return image.updateMask(mask)

    def test_EEBatchToDrive(self, eeImageGeometry, imageParams,satUsed,GDrive,pathPrefix=""):
        uf_reef_geom = eeImageGeometry
        print('geom=', eeImageGeometry)
        print('centroid=', eeImageGeometry.centroid())
        #landsat = ee.ImageCollection("LANDSAT/LC08/C01/T1").filterDate('2016-01-01', '2017-01-01').filterBounds(uf_reef_geom)
        #target_lat_lon = np.array([ center_latitude, center_longitude])
        #uf_reef_geom = ee.Geometry.Point(target_lat_lon.tolist())
        #uf_reef_geom2 = ee.Geometry.LineString([[-120, 35], [-119, 35], [-119, 34], [-120, 34]])
        
        # LIST OF SATELLITES
        # Planet SkySat Public Ortho Imagery, RGB
        # 0-255 RGB pixels, 0.8m resolution
        #sat = 'skysat' 
        # NAIP: National Agriculture Imagery Program
        # 0-255 RGB pixels, 1m resolution
        if (satUsed == "naip"):
            sat = 'usda_naip' 
        # USGS Landsat 8 Surface Reflectance Tier 1 
        # 0-10000 pixel values 30m resolution
        #sat = 'landsat-8-sr' 
        # Sentinel-1 C-band Synthetic Aperture Radar Ground Range corrected
        # -50 to 1 db range resolution
        if (satUsed == "sar"):
            sat = 'sentinel-1-grd' 
        # Sentinel-2 orthorectified atmospherically corrected surface reflectance.
        # pixel range 10000 resolution 10m
        #sat = 'sentinel-2-sr' 
        if (satUsed == "cloudy"):
            sat = 'sentinel-2-sr-cloudy'
        if (satUsed == "cloudless"):
            sat = 'sentinel-2-sr-cloudless'
        if (sat == 'skysat'):
            imgCollection = ee.ImageCollection('SKYSAT/GEN-A/PUBLIC/ORTHO/RGB') \
                                            .select(['R', 'G', 'B']) \
                                            .filterBounds(uf_reef_geom)
            # Define the visualization parameters.
            vizParams = {
               'min': 0.0,
               'max': 255.0
            }
        elif (sat == 'usda_naip'):
            imgCollection = ee.ImageCollection('USDA/NAIP/DOQQ') \
                                            .select(['R', 'G', 'B']) \
                                            .filterBounds(uf_reef_geom) \
                                            .filter(ee.Filter.date('2017-01-01', '2018-12-31'))
            # Define the visualization parameters.
            vizParams = {
               'min': 0.0,
               'max': 255.0
            }
        elif (sat == 'sentinel-2-sr-cloudless'):
            imgCollection = ee.ImageCollection('COPERNICUS/S2_SR') \
                                            .filterBounds(uf_reef_geom) \
                                            .filterDate('2018-06-01', '2021-01-30') \
                                            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',5)) \
                                            .map(self.maskS2clouds)
            imgCollection = imgCollection.select(['TCI_R', 'TCI_G', 'TCI_B'])
                                            #.filter(ee.Filter.rangeContains('CLOUD_COVERAGE_ASSESSMENT', 35,60))      
            # Define the visualization parameters.
            vizParams = {
               'bands': ['TCI_R', 'TCI_G', 'TCI_B'],
               'min': 0,
               'max': 255 # range is 10000 
            }
        elif (sat == 'sentinel-2-sr-cloudy'):
            imgCollection = ee.ImageCollection('COPERNICUS/S2_SR') \
                                            .filterBounds(uf_reef_geom) \
                                            .filterDate('2018-06-01', '2021-01-30') \
                                            .filter(ee.Filter.gt('CLOUDY_PIXEL_PERCENTAGE',5)) \
                                            .map(self.maskS2clouds)
            imgCollection = imgCollection.select(['TCI_R', 'TCI_G', 'TCI_B'])                                      
            # Define the visualization parameters.
            vizParams = {
               'bands': ['TCI_R', 'TCI_G', 'TCI_B'],
               'min': 0,
               'max': 255 # range is 10000 
            }
        elif (sat == 'sentinel-1-grd'):
            imgCollection = ee.ImageCollection('COPERNICUS/S1_GRD') \
                                            .filterBounds(uf_reef_geom) \
                                            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
                                            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
                                            .select('VV')
                                            #.map(function22(image, geom))
#                                            .map(function(image) {
#                                                edge = image.lt(-30.0);
#                                                maskedImage = image.mask().and(edge.not());
#                                                return image.updateMask(maskedImage);
#                                                })
            # Define the visualization parameters.
            vizParams = {
               'bands': ['VV'],
               'min': -25,  #-50
               'max': 2     # 1
            }
        else:
            imgCollection = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR') \
                                            .select(['B4', 'B3', 'B2']) \
                                            .filterBounds(uf_reef_geom) \
                                            .sort('CLOUD_COVER', False)  
                    #.filterDate('2016-01-01', '2017-01-01') \
            # Define the visualization parameters.
            vizParams = {
                'bands': ['B4', 'B3', 'B2'],
                'min': 0,
                'max': 3000,
                'gamma': 1.4
            }
        #reduced_image = imgCollection.first()
        #reduced_image = imgCollection.median()
        reduced_image = imgCollection.mean()
        scaleVal = reduced_image.projection().nominalScale()
        print('projection ',str(reduced_image.projection().getInfo()))
        print('Centroid coordinates = ',str(eeImageGeometry.centroid().getInfo()['coordinates']))
        print('Nominal scale = ',str(scaleVal.getInfo()))
        print('EPSG 4326 Rect Coordinates = ',str(eeImageGeometry.getInfo()['coordinates']))
        centroid = eeImageGeometry.centroid().getInfo()['coordinates']
        filenameStr = 'sat_'+ sat + '_' + str(centroid[1]) + 'N' +'_' + str(centroid[0]) + 'E'+ '_' + str(scaleVal.getInfo()) + 'm'
        print('Filename = ', filenameStr)
        #scaleVal = scaleVal.getInfo()
        scaleVal = 10
        # CHANGE BETWEEN USING DIMENSIONS AND SCALE FOR NAIP AND SAR/Sen-2 images respectively
        if (satUsed == "naip"):
            task_config = {
                'fileFormat': 'GeoTIFF',
                #'scale': scaleVal, # m/pixel ?
                'maxPixels' : 6e9,
                'region' : eeImageGeometry.getInfo()['coordinates'],
                'dimensions' : str(imageParams['xy_dimensions_pixels'][0]) + "x" + str(imageParams['xy_dimensions_pixels'][1]),
                #'crs': 'EPSG:3857'
                'crs': 'EPSG:4326'
            }
        else:
            task_config = {
                'fileFormat': 'GeoTIFF',
                'scale': scaleVal, # m/pixel ?
                'maxPixels' : 6e9,
                'region' : eeImageGeometry.getInfo()['coordinates'],
                #'dimensions' : str(imageParams['xy_dimensions_pixels'][0]) + "x" + str(imageParams['xy_dimensions_pixels'][1]),
                #'crs': 'EPSG:3857'
                'crs': 'EPSG:4326'
            }
        listOfImages = imgCollection.toList(10);
        landsat_image = ee.Image(listOfImages.get(0))
#        urlpath = landsat_image.getDownloadUrl({
#                    'scale': 60,
#                    'crs': 'EPSG:4326',
#                    'region': '[[-120, 35], [-119, 35], [-119, 34], [-120, 34]]'
#                    })
        #print(urlpath)
        
        #r = requests.get(urlpath, allow_redirects=True)
        #print(r.headers.get('content-type'))
        #open('download.zip', 'wb').write(r.content)

        # display the mape
        #if (sat == 'usda_naip'):
        #    ee.mapclient.centerMap(-86.572972, 30.474693, 15)
        #else:
        #Map.addLayer(your_data.clip(roi), {}, "Your_data_name", true);
        ee.mapclient.centerMap(float(centroid[0]), float(centroid[1]), 14)           
        #ee.mapclient.centerMap(-86.572972, 30.474693) 
        ee.mapclient.addToMap(imgCollection, vizParams, sat)        
        #task = ee.batch.Export.image.toDrive(landsat, 'test', **task_config)
        # task_config fields
        #        image: The image to be exported.
        #        description: Human-readable name of the task.
        #        folder: The name of a unique folder in your Drive account to
        #            export into. Defaults to the root of the drive.
        #        fileNamePrefix: The Google Drive filename for the export.
        #            Defaults to the name of the task.
        #        dimensions: The dimensions of the exported image. Takes either a
        #            single positive integer as the maximum dimension or "WIDTHxHEIGHT"
        #            where WIDTH and HEIGHT are each positive integers.
        #        region: The lon,lat coordinates for a LinearRing or Polygon
        #            specifying the region to export. Can be specified as a nested
        #            lists of numbers or a serialized string. Defaults to the image's
        #            region.
        #        scale: The resolution in meters per pixel. Defaults to the
        #            native resolution of the image assset unless a crsTransform
        #            is specified.
        #        crs: The coordinate reference system of the exported image's
        #            projection. Defaults to the image's default projection.
        #        crsTransform: A comma-separated string of 6 numbers describing
        #            the affine transform of the coordinate reference system of the
        #            exported image's projection, in the order: xScale, xShearing,
        #            xTranslation, yShearing, yScale and yTranslation. Defaults to
        #            the image's native CRS transform.
        #        maxPixels: The maximum allowed number of pixels in the exported
        #            image. The task will fail if the exported region covers more
        #            pixels in the specified projection. Defaults to 100,000,000.
        #        shardSize: Size in pixels of the shards in which this image will be
        #            computed. Defaults to 256.
        #        fileDimensions: The dimensions in pixels of each image file, if the
        #            image is too large to fit in a single file. May specify a
        #            single number to indicate a square shape, or a tuple of two
        #            dimensions to indicate (width,height). Note that the image will
        #            still be clipped to the overall image dimensions. Must be a
        #            multiple of shardSize.
        #        skipEmptyTiles: If true, skip writing empty (i.e. fully-masked)
        #            image tiles. Defaults to false.
        #        fileFormat: The string file format to which the image is exported.
        #            Currently only 'GeoTIFF' and 'TFRecord' are supported, defaults to
        #            'GeoTIFF'.
        #        formatOptions: A dictionary of string keys to format specific options.

        task = ee.batch.Export.image.toDrive(reduced_image, filenameStr, **task_config)
#        task_config = {
#            'image': landsat_image,
#            'description': 'imageToDriveExample',
#            'scale': 30,
#            'region': eeImageGeometry
#        }
#        task = ee.batch.Export.image.toDrive(**task_config)

        task.start()
#        ee.batch.data.startProcessing(task.id, task.config)
        # Note: I also tried task.start() instead of this last line but the problem is the same, task completed, no file created. 

#        task = ee.batch.Export.image.toDrive(image=my_img,
#                                     region=my_geometry.getInfo()['coordinates'],
#                                     description='my_description',
#                                     folder='my_gdrive_folder',
#                                     fileNamePrefix='my_file_prefix',
#                                     scale=my_scale,
#                                     crs=my_crs)
        # Printing the task list successively 

        while(task.status()['state'] == 'READY' or task.status()['state'] == 'RUNNING'):
            tasks = ee.batch.Task.list()
            print(tasks)
            print(task.status())
            time.sleep(5)        
        print(task.status()['state'])
        
        return filenameStr
        
    def driveDownloader(self,fileBatch):
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()
        GDrive = GoogleDrive(gauth)
    
        qID = None
        file_list = GDrive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
        for file in fileBatch:
            pathPrefix = file[0]
            filenameStr = file[1]
            satUsed = file[2]
            for file1 in file_list:
                if(file1['title'] == filenameStr+'.tif'):
                    qID = file1['id']

            if (qID is not None):        
                file = GDrive.CreateFile({'id': qID})
                file.GetContentFile(pathPrefix+filenameStr+'.tif')

                src = gdal.Open(pathPrefix+filenameStr+'.tif')
                if(satUsed == 'naip' or satUsed == 'cloudy' or satUsed == 'cloudless'):
                    gdal.Translate(pathPrefix+filenameStr+'.png',src,format='PNG')
                elif(satUsed == 'sar'):
                    band = src.GetRasterBand(1)
                    (bMin,bMax) = band.ComputeRasterMinMax(True)
                    gdal.Translate(pathPrefix+filenameStr+'.png',src,format='PNG',scaleParams=[[bMin,bMax]])
                src = None
        return

if __name__ == '__main__':
    #try:
    if (len(sys.argv) == 1 or not (sys.argv[1] == 'random' or sys.argv[1] == 'manual' or sys.argv[1] == 'randomMulti')):
        print("First input must be \"manual\", \"random\", or \"randomMulti\" (Without the quotations)")
    elif (len(sys.argv) < 9 and sys.argv[1] == 'random'):
        print("Need arguments with:\nThe upper left corner lat long coordinates being the first 2 arguments.\nThe bottom right corner lat long coordinates for the next 2 arguments.\nThe number of pixels in x and y.\nWhat meter/pixel ratio is needed.\nEXAMPLE: python googleearth_client.py random 48.852411 -121.706750 32.534790 -80.935663 1920 1080 1")
    elif (len(sys.argv) < 10 and sys.argv[1] == 'randomMulti'):
        print("Need arguments with:\nThe upper left corner lat long coordinates being the first 2 arguments.\nThe bottom right corner lat long coordinates for the next 2 arguments.\nThe number of pixels in x and y.\nWhat meter/pixel ratio is needed.\nHow many images are needed.\nEXAMPLE: python googleearth_client.py randomMulti 48.852411 -121.706750 32.534790 -80.935663 1920 1080 1 50")
    elif (len(sys.argv) < 7 and sys.argv[1] == 'manual'):
        print("Need arguments with:\nThe lat long coordinates being the first 2 arguments.\nThe number of pixels in x and y.\nWhat meter/pixel ratio is needed.\nEXAMPLE: python googleearth_client.py manual 48.852411 -121.706750 1920 1080 1")
    else:
        geClient = GoogleEarthClient(sys.argv)
    #except:
    #    pass

# Google Maps utilities including a way to download top view images 
# given coordinates. Also a way to transform pixel distances to ground plane coordinates
# see below for gmaps library documantation
# https://github.com/googlemaps/google-maps-services-python/blob/master/googlemaps/maps.py
# see below for transfering pixel to meter
# https://stackoverflow.com/questions/9356724/google-map-api-zoom-range

API_KEY = "AIzaSyB7ibnkK0H6FRroqEKgP55SajuqnUFEo0Y"
from Libs import *

def download_image(center, file_name):
    gmaps = googlemaps.Client(key=API_KEY)
    gen = gmaps.static_map(size=(640, 640),
               center=center, zoom=19, 
               format="png", maptype="hybrid")
    with open(file_name, "wb") as f:
        for chunk in gen:
            if chunk: f.write(chunk)

def meter_per_pixel(coordinates, zoom=19):
    m_per_p = 156543.03392 * np.cos(coordinates[0] * np.pi / 180) / np.pow(2, zoom)
    return m_per_p
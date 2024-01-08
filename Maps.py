# Google Maps utilities including a way to download top view images 
# given coordinates. Also a way to transform pixel distances to ground plane coordinates
# see below for gmaps library documantation
# https://github.com/googlemaps/google-maps-services-python/blob/master/googlemaps/maps.py
# see below for transfering pixel to meter
# https://stackoverflow.com/questions/9356724/google-map-api-zoom-range

API_KEY = "AIzaSyB7ibnkK0H6FRroqEKgP55SajuqnUFEo0Y"
from Libs import *
from Utils import *

def download_image(center, file_name):
    gmaps = googlemaps.Client(key=API_KEY)
    gen = gmaps.static_map(size=(640, 640),
               center=center, zoom=19, 
               format="png", maptype="hybrid")
    with open(file_name, "wb") as f:
        for chunk in gen:
            if chunk: f.write(chunk)

def meter_per_pixel(center, zoom=19):
    m_per_p = 156543.03392 * np.cos(center[0] * np.pi / 180) / np.power(2, zoom)
    return m_per_p

def pix2meter(args):
    # change two files
    # values should be stored accordingly
    # 1. reprojected tracks: args.ReprojectedPkl -> args.ReprojectedPklMeter
    r = meter_per_pixel(args.MetaData['center'])
    if os.path.exists(args.ReprojectedPkl):
        df = pd.read_pickle(args.ReprojectedPkl)
        scale_function = lambda x: r*x
        df['x'] = df['x'].apply(scale_function)
        df['y'] = df['y'].apply(scale_function)
        df.to_pickle(args.ReprojectedPklMeter)

    # 2. labeled trackes: args.TrackLabellingExportPth -> args.TrackLabellingExportPthMeter
    if os.path.exists(args.TrackLabellingExportPth):
        df = pd.read_pickle(args.TrackLabellingExportPth)
        scale_function = lambda x: np.array([[r*xi[0],r*xi[1]] for xi in x])
        df['trajectory'] = df['trajectory'].apply(scale_function)
        df.to_pickle(args.TrackLabellingExportPthMeter)

    # # show the scale with roi
    # print(r)
    # # print(np.array(args.MetaData["roi"]))

    # M = np.load(args.HomographyNPY, allow_pickle=True)[0]
    # roi_rep = []
    # roi = args.MetaData["roi"]
    # for p in roi:
    #     point = np.array([p[0], p[1], 1])
    #     new_point = M.dot(point)
    #     new_point /= new_point[2]
    #     roi_rep.append([new_point[0], new_point[1]])
    # roi_rep = np.array(roi_rep)
    # print(roi_rep)
    # print(r*roi_rep)

    return SucLog("pixel cordinates changed to meter cordinates")



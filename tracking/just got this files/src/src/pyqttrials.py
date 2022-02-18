import json

from PyQt5 import QtCore, QtWidgets, QtWebEngineWidgets, QtWebChannel, QtNetwork
from PyQt5.QtWidgets import *

import cv2
import math
import numpy as np
import os
import utm

from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

HTML = '''
<!DOCTYPE html>
<html>
<body>
<br>
<br>
<br>
<h1>Camera extrinsics calibration tool</h1>
<br>
<br>
<br>
<div>
<div style="width:49%;float:left" clear="left">
  <input type="text" placeholder="Type something..." id="myInput">
  <button type="button" onclick="getInputValue();">Search</button>
  <div id="googleMap" style="height:600px;"></div>
  <label id="coordinates"></label>
</div>
<div style="width:49%;float:right" clear="right">
	<input type="file" onchange="previewFile()"><br>
  <img src="" width="796;" height="540;" alt="Image preview..." id="localimg" onclick="point_it(event);">
  <canvas width="796;" height="540;" id="imgcanvas" onclick="point_it(event);"></canvas><br>
	<label id="impixels"></label>
</div>
</div>

<script>
var coordinates_nums = [];
var impixels_nums = [];
//////////////////////////////////////////////////////////////////////////////////////////////
function previewFile() {
  
    // Where you will display your image
    //var preview = document.querySelector('img');
	var preview = document.getElementById("localimg");
  var c = document.getElementById("imgcanvas");
  c.style.position = "absolute";
  c.style.left = preview.offsetLeft+"px";
  c.style.top = preview.offsetTop+"px";
    // The button where the user chooses the local image to display
    var file = document.querySelector('input[type=file]').files[0];
    // FileReader instance
    var reader  = new FileReader();

    // When the image is loaded we will set it as source of
    // our img tag
    reader.onloadend = function () {
      preview.src = reader.result;
    }

    
    if (file) {
      // Load image as a base64 encoded URI
      reader.readAsDataURL(file);
    } else {
      preview.src = "";
    }
  }
  
//////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////
var impixels=""
function point_it(event) {
  var c = document.getElementById("imgcanvas");
  var ctx = c.getContext("2d");
  
	pos_x = event.offsetX ? (event.offsetX) : event.pageX - document.getElementById("localimg").offsetLeft;
	pos_y = event.offsetY ? (event.offsetY) : event.pageY - document.getElementById("localimg").offsetTop;
	if (impixels == "") {
      impixels = '('+pos_x+', '+ pos_y+')';
  }
  else {
    impixels = impixels + ',<br/>' + '('+pos_x+', '+ pos_y+')';
  }
  document.getElementById('impixels').innerHTML = impixels;
  ctx.beginPath();
  ctx.arc(pos_x, pos_y, 5, 0, Math.PI * 2, true);
  ctx.lineWidth = 3;
  ctx.stroke();
  impixels_nums.push([pos_x, pos_y])
}

//////////////////////////////////////////////////////////////////////////////////////////////
var address = "Essa Road Gowan Street"
var coordinates = ""
var elevations = ""
var markertitle = 1
function getInputValue(){
          // Selecting the input element and get its value 
          var inputVal = document.getElementById("myInput").value;
          
          // Displaying the value
          //alert(inputVal);
          address = inputVal;
          myMap();
      }
function myMap() {
geocoder = new google.maps.Geocoder();
var mapProp= {
	mapTypeId: 'satellite',
  zoom:20,
};
var map = new google.maps.Map(document.getElementById("googleMap"),mapProp);
map.setTilt(0);
if (geocoder) {
    geocoder.geocode({
      'address': address
    }, function(results, status) {
      if (status == google.maps.GeocoderStatus.OK) {
        if (status != google.maps.GeocoderStatus.ZERO_RESULTS) {
          map.setCenter(results[0].geometry.location);

          var infowindow = new google.maps.InfoWindow({
            content: '<b>' + address + '</b>',
            //size: new google.maps.Size(150, 50)
          });

        } else {
          alert("No results found");
        }
      } else {
        alert("Geocode was not successful for the following reason: " + status);
      }
    });
	}
var elevator = new google.maps.ElevationService;
google.maps.event.addListener(map, 'click', function(event) {

    marker = new google.maps.Marker({position: event.latLng, map: map,label:markertitle.toString()});
    markertitle = markertitle+1;
    if (coordinates == "") {
      coordinates = event.latLng;
    }
    else {
      coordinates = coordinates + ',<br/>' + event.latLng;
    }
    document.getElementById('coordinates').innerHTML = coordinates;
    coordinates_nums.push([event.latLng.lat(), event.latLng.lng()]);
    //alert(event.latLng.lat())
	  //alert(event.latLng);
});
}
//////////////////////////////////////////////////////////////////////////////////////////////////
function runCalibScript(){
  return [coordinates_nums, impixels_nums];
}
</script>
<script src="https://maps.googleapis.com/maps/api/js?key=AIzaSyAdh4Lum-3V_XyZemkjY6pCiLUi3Skg9M8&callback=myMap"></script>

</body>
</html>
'''

def all_collector(all_coords):
    latlng_pairs = all_coords[0]
    listPt = all_coords[1]

    r = 6371000 # meters
    phi_0 = latlng_pairs[0][1]
    cos_phi_0 = math.cos(math.radians(phi_0))

    def to_xy(point, r, cos_phi_0):
        lam = point[0]
        phi = point[1]
        return (r * math.radians(lam) * cos_phi_0, r * math.radians(phi))

    utmx = []
    utmy = []
    for point in latlng_pairs:
        x, y, z, zl = utm.from_latlon(point[0], point[1])
        utmx.append(x)
        utmy.append(y)

    world_points_3d = [[x[0],x[1],200] for x in zip(utmx,utmy)]
    intrinsics = np.loadtxt(curr_folder+'/cameraParams/intrinsics.txt')
    intrinsics = intrinsics.transpose()
    distortion = np.loadtxt(curr_folder+'/cameraParams/distortion.txt')
    [retval, rvec, tvec] = cv2.solvePnP(np.array(world_points_3d, dtype='float32'), np.array(listPt, dtype='float32'), intrinsics, distortion)
    [rvec3, jacobian] = cv2.Rodrigues(rvec)

    #rvec3_file_read = np.genfromtxt('../data/' + folder_name + '/'+'rvec3.csv', delimiter=',')
    cam_location = np.matmul(np.linalg.inv(-rvec3),(tvec))
    roll = math.atan2(-rvec3[2][1], rvec3[2][2])
    pitch = math.asin(rvec3[2][0])
    yaw = math.atan2(-rvec3[1][0], rvec3[0][0])
    cam_height = cam_location[2]
    #cv2.circle(image, (cam_location[0],cam_location[1]), 15, (0, 0, 255), 3)
    print("cam location = {}, {}, roll = {}, pitch = {}, yaw = {}, cam height = {}".format(cam_location[0], cam_location[1],roll*180/math.pi, pitch*180/math.pi, yaw*180/math.pi, cam_height))



class GeoCoder(QtNetwork.QNetworkAccessManager):
    class NotFoundError(Exception):
        pass

    def geocode(self, location, api_key):
        url = QtCore.QUrl("https://maps.googleapis.com/maps/api/geocode/xml")

        query = QtCore.QUrlQuery()
        query.addQueryItem("key", api_key)
        query.addQueryItem("address", location)
        url.setQuery(query)
        request = QtNetwork.QNetworkRequest(url)
        reply = self.get(request)
        loop = QtCore.QEventLoop()
        reply.finished.connect(loop.quit)
        loop.exec_()
        reply.deleteLater()
        self.deleteLater()
        return self._parseResult(reply)

    def _parseResult(self, reply):
        xml = reply.readAll()
        reader = QtCore.QXmlStreamReader(xml)
        while not reader.atEnd():
            reader.readNext()
            if reader.name() != "geometry": continue
            reader.readNextStartElement()
            if reader.name() != "location": continue
            reader.readNextStartElement()
            if reader.name() != "lat": continue
            latitude = float(reader.readElementText())
            reader.readNextStartElement()
            if reader.name() != "lng": continue
            longitude = float(reader.readElementText())
            return latitude, longitude
        raise GeoCoder.NotFoundError


class QGoogleMap(QtWebEngineWidgets.QWebEngineView):
    mapMoved = QtCore.pyqtSignal(float, float)
    mapClicked = QtCore.pyqtSignal(float, float)
    mapRightClicked = QtCore.pyqtSignal(float, float)
    mapDoubleClicked = QtCore.pyqtSignal(float, float)

    markerMoved = QtCore.pyqtSignal(str, float, float)
    markerClicked = QtCore.pyqtSignal(str, float, float)
    markerDoubleClicked = QtCore.pyqtSignal(str, float, float)
    markerRightClicked = QtCore.pyqtSignal(str, float, float)
    coord_px = []

    def __init__(self, api_key, parent=None):
        super(QGoogleMap, self).__init__(parent)
        self._api_key = api_key
        channel = QtWebChannel.QWebChannel(self)
        self.page().setWebChannel(channel)
        channel.registerObject("qGoogleMap", self)
        #self.page().runJavaScript(JS)

        html = HTML.replace("API_KEY", "YOUR_API_KEY_HERE")
        self.setHtml(html)
        self.loadFinished.connect(self.on_loadFinished)
        self.initialized = False

        self._manager = QtNetwork.QNetworkAccessManager(self)
        b1 = QPushButton(self)
        b1.setText("Run calib")
        b1.clicked.connect(self.b1_clicked)

    
    def b1_clicked(self):
        self.runScript("runCalibScript()",all_collector)
        #print(self.coord_px)

    @QtCore.pyqtSlot()
    def on_loadFinished(self):
        self.initialized = True

    def waitUntilReady(self):
        if not self.initialized:
            loop = QtCore.QEventLoop()
            self.loadFinished.connect(loop.quit)
            loop.exec_()

    def geocode(self, location):
        return GeoCoder(self).geocode(location, self._api_key)

    def centerAtAddress(self, location):
        try:
            latitude, longitude = self.geocode(location)
        except GeoCoder.NotFoundError:
            print("Not found {}".format(location))
            return None, None
        self.centerAt(latitude, longitude)
        return latitude, longitude

    def addMarkerAtAddress(self, location, **extra):
        if 'title' not in extra:
            extra['title'] = location
        try:
            latitude, longitude = self.geocode(location)
        except GeoCoder.NotFoundError:
            return None
        return self.addMarker(location, latitude, longitude, **extra)

    @QtCore.pyqtSlot(float, float)
    def mapIsMoved(self, lat, lng):
        self.mapMoved.emit(lat, lng)

    @QtCore.pyqtSlot(float, float)
    def mapIsClicked(self, lat, lng):
        self.mapClicked.emit(lat, lng)

    @QtCore.pyqtSlot(float, float)
    def mapIsRightClicked(self, lat, lng):
        self.mapRightClicked.emit(lat, lng)

    @QtCore.pyqtSlot(float, float)
    def mapIsDoubleClicked(self, lat, lng):
        self.mapDoubleClicked.emit(lat, lng)

    # markers
    @QtCore.pyqtSlot(str, float, float)
    def markerIsMoved(self, key, lat, lng):
        self.markerMoved.emit(key, lat, lng)

    @QtCore.pyqtSlot(str, float, float)
    def markerIsClicked(self, key, lat, lng):
        self.markerClicked.emit(key, lat, lng)

    @QtCore.pyqtSlot(str, float, float)
    def markerIsRightClicked(self, key, lat, lng):
        self.markerRightClicked.emit(key, lat, lng)

    @QtCore.pyqtSlot(str, float, float)
    def markerIsDoubleClicked(self, key, lat, lng):
        self.markerDoubleClicked.emit(key, lat, lng)

    def runScript(self, script, callback=None):
        if callback is None:
            self.page().runJavaScript(script)
        else:
            self.page().runJavaScript(script, callback)

    def centerAt(self, latitude, longitude):
        self.runScript("gmap_setCenter({},{})".format(latitude, longitude))

    def center(self):
        self._center = {}
        loop = QtCore.QEventLoop()

        def callback(*args):
            self._center = tuple(args[0])
            loop.quit()

        self.runScript("gmap_getCenter()", callback)
        loop.exec_()
        return self._center

    def setZoom(self, zoom):
        self.runScript("gmap_setZoom({})".format(zoom))

    def addMarker(self, key, latitude, longitude, **extra):
        return self.runScript(
            "gmap_addMarker("
            "key={!r}, "
            "latitude={}, "
            "longitude={}, "
            "{}"
            "); ".format(key, latitude, longitude, json.dumps(extra)))

    def moveMarker(self, key, latitude, longitude):
        return self.runScript(
            "gmap_moveMarker({!r}, {}, {});".format(key, latitude, longitude))

    def setMarkerOptions(self, keys, **extra):
        return self.runScript(
            "gmap_changeMarker("
            "key={!r}, "
            "{}"
            "); "
                .format(keys, json.dumps(extra)))

    def deleteMarker(self, key):
        return self.runScript(
            "gmap_deleteMarker("
            "key={!r} "
            "); ".format(key))


if __name__ == '__main__':
    import sys

    API_KEY = "AIzaSyAdh4Lum-3V_XyZemkjY6pCiLUi3Skg9M8"

    app = QtWidgets.QApplication(sys.argv)
    #win = QDialog()
    
    #win.show()
    w = QGoogleMap(api_key=API_KEY)
    
    #w.resize(640, 480)
    w.show()
    w.waitUntilReady()
    # w.setZoom(14)
    # lat, lng = w.centerAtAddress("Lima Peru")
    # if lat is None and lng is None:
    #     lat, lng = -12.0463731, -77.042754
    #     w.centerAt(lat, lng)

    # w.addMarker("MyDragableMark", lat, lng, **dict(
    #     icon="http://maps.gstatic.com/mapfiles/ridefinder-images/mm_20_red.png",
    #     draggable=True,
    #     title="Move me!"
    # ))

    # for place in ["Plaza Ramon Castilla", "Plaza San Martin", ]:
    #     w.addMarkerAtAddress(place, icon="http://maps.gstatic.com/mapfiles/ridefinder-images/mm_20_gray.png")

    # w.mapMoved.connect(print)
    # w.mapClicked.connect(print)
    # w.mapRightClicked.connect(print)
    # w.mapDoubleClicked.connect(print)
    sys.exit(app.exec_())
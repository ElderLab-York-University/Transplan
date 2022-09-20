# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """
    M, R, Mi = None, None, None
    def __init__(self, tlwh, confidence, feature, M, R):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)
        self.update_MRMi(M, R)

    def update_MRMi(self, M, R):
        if self.M is None:
            self.M = M
            self.R = R
            self.Mi = np.linalg.inv(M)

    @staticmethod
    def init_MRMi(M, R):
        Detection.M = M
        Detection.R = R
        Detection.Mi = np.linalg.inv(M)

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        return self.tlwh_to_xyah(self.tlwh)
    
    ##!! back projection happens here
    ##!! make sure to use these conversion tools in the tracker.py as well
    @staticmethod
    def tlwh_to_xyah(tlwh):
        x_cp, y_cp, w, h  = Detection.tlwh_to_cpwh(tlwh)
        a = w/h
        x_gc , y_gc = Detection.to_ground_cor(x_cp, y_cp)
        return np.array([x_gc, y_gc, a, h])

    @staticmethod
    def xyah_to_tlwh(xyah):
        x_gc, y_gc, a, h = xyah
        x_cp, y_cp = Detection.to_img_cor(x_gc, y_gc)
        w = a*h
        cpwh = np.array([x_cp, y_cp, w, h])
        return Detection.cpwh_to_tlwh(cpwh)

    @staticmethod
    def tlwh_to_cpwh(tlwh):
        ''' converts self.tlwh to contact point in the image plane
        '''
        x_center , y_center, w , h = tlwh
        x_cp = x_center + w/2
        y_cp  = y_center
        return np.array([x_cp, y_cp, w, h])
   
    @staticmethod
    def cpwh_to_tlwh(cpwh):
        x_cp, y_cp, w, h = cpwh
        x_center = x_cp - w/2
        y_center = y_cp
        return np.array([x_center, y_center, w, h])

    @staticmethod
    def to_ground_cor(x, y):
        point = np.array([x, y, 1])
        new_point = Detection.M.dot(point)
        new_point /= new_point[2]
        x_g , y_g = new_point[0], new_point[1]
        return x_g, y_g

    @staticmethod
    def to_img_cor(x, y):
        point = np.array([x, y, 1])
        new_point = Detection.Mi.dot(point)
        new_point /= new_point[2]
        x_i , y_i = new_point[0], new_point[1]
        return x_i, y_i

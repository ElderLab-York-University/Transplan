import cv2
import numpy as np


class backend():
    fileName1 = ""
    fileName2 = ""
    img1 = []
    img2 = []
    symimg_filename = ""
    symimg = []

    def setFileName1(self,_fileName):
        self.fileName1 = _fileName
        self.img1 = cv2.imread(self.fileName1)
    
    def setFileName2(self,_fileName):
        self.fileName2 = _fileName
        self.img2 = cv2.imread(self.fileName2)

    def doHomography(self, pts1, pts2):
        cv2hom = cv2.findHomography(np.array(pts1), np.array(pts2))
        np.save('homography_'+str(len(pts1))+'.npy',cv2hom)
        np.savetxt('homography_'+str(len(pts1))+'.txt',cv2hom[0], delimiter=',')
        warped_image = cv2.warpPerspective(self.img1.copy(), cv2hom[0], (self.img2.shape[1],self.img2.shape[0]))
        alpha = 0.7
        beta = (1.0 - alpha)
        dst = cv2.addWeighted(warped_image, alpha, self.img2, beta, 0.0)
        cv2.imshow('warped',dst)
        return True
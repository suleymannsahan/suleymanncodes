import cv2
import numpy as np

class FiltersTransform:
    def __init__(self, frame=None, lower_white=(0, 190, 0), upper_white=(255, 255, 255), white_mask=None, mask=None,
                 contours=None, filtered_frame=None, area=0, sobelx=None, sobely=None, sobel_combined=None, threshold=None):
        self.frame = frame
        self.lower_white = lower_white
        self.upper_white = upper_white
        self.white_mask = white_mask
        self.mask = mask
        self.contours = contours
        self.filtered_frame = filtered_frame
        self.area = area
        self.sobelx = sobelx
        self.sobely = sobely
        self.sobel_combined = sobel_combined
        self.threshold = threshold

    def convert_to_hls(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    
    def get_white_mask(self, hls_frame):
        self.white_mask = cv2.inRange(hls_frame, self.lower_white, self.upper_white)
        return self.white_mask
    
    def findContours(self, frame):
        self.mask = np.zeros_like(frame)
        self.contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in self.contours:
            self.area = cv2.contourArea(contour)
            if self.area > 50:
                cv2.drawContours(self.mask, [contour], -1, 255, thickness=cv2.FILLED)
        self.filtered_frame = cv2.bitwise_and(frame, self.mask)
        return self.filtered_frame
    
    def applying_blur(self, gray_frame):
        return cv2.GaussianBlur(gray_frame, (5, 5), 0)
    
    def applying_canny(self, gray_frame):
        return cv2.Canny(gray_frame, 100, 200)

    def applying_sobel(self, gray_frame):
        self.sobelx = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
        self.sobely = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
        self.sobel_combined = cv2.magnitude(self.sobelx, self.sobely)
        return self.sobel_combined

    def applying_threshold(self, gray_frame):
        _, self.threshold = cv2.threshold(gray_frame, 220, 255, cv2.THRESH_BINARY)
        return self.threshold
    
    def getting_matrix(self, points1, points2):
        return cv2.getPerspectiveTransform(points1, points2)
    
    def transform(self, frame, matrix, dsize):
        return cv2.warpPerspective(frame, matrix, dsize)

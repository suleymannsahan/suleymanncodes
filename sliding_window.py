import cv2
import numpy as np

class HistogramSlidingWindow:
    def __init__(self, window_height=40):
        self.window_height = window_height
        self.lx, self.mx, self.rx = [], [], []
        
    def applying_histogram(self, threshold):
        self.histogram = np.sum(threshold[threshold.shape[0] // 2 :, :], axis=0)
        self.mid_point = self.histogram.shape[0] // 2
        self.leftx_base = np.argmax(self.histogram[:self.mid_point // 2])  # Sol şerit
        self.middlex_base = np.argmax(self.histogram[self.mid_point // 2 : self.mid_point]) + self.mid_point // 2  # Orta şerit
        self.rightx_base = np.argmax(self.histogram[self.mid_point :]) + self.mid_point  # Sağ şerit
        return self.histogram, self.leftx_base, self.middlex_base, self.rightx_base
    
    def sliding_window(self, threshold, leftx_base, middlex_base, rightx_base):
        self.y = threshold.shape[0]
        self.thrs = threshold.copy()
        
        while self.y > 0:
            # Sol şerit penceresi
            self.left_img = self.thrs[self.y - self.window_height:self.y, max(0, leftx_base - 50):min(self.thrs.shape[1], leftx_base + 50)]
            self.contours, _ = cv2.findContours(self.left_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in self.contours:
                self.M = cv2.moments(contour)
                if self.M["m00"] != 0:
                    self.cx = int(self.M["m10"] / self.M["m00"])
                    leftx_base = leftx_base - 50 + self.cx
                    self.lx.append(leftx_base)

            # Orta şerit penceresi
            self.middle_img = self.thrs[self.y - self.window_height:self.y, max(0, middlex_base - 50):min(self.thrs.shape[1], middlex_base + 50)]
            self.contours, _ = cv2.findContours(self.middle_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in self.contours:
                self.M = cv2.moments(contour)
                if self.M["m00"] != 0:
                    self.cx = int(self.M["m10"] / self.M["m00"])
                    middlex_base = middlex_base - 50 + self.cx
                    self.mx.append(middlex_base)

            # Sağ şerit penceresi
            right_img = self.thrs[self.y - self.window_height:self.y, max(0, rightx_base - 50):min(self.thrs.shape[1], rightx_base + 50)]
            contours, _ = cv2.findContours(right_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                self.M = cv2.moments(contour)
                if self.M["m00"] != 0:
                    self.cx = int(self.M["m10"] / self.M["m00"])
                    rightx_base = rightx_base - 50 + self.cx
                    self.rx.append(rightx_base)

            cv2.rectangle(self.thrs, (leftx_base - 50, self.y), (leftx_base + 50, self.y - self.window_height), (255, 255, 255), 2)
            cv2.rectangle(self.thrs, (middlex_base - 50, self.y), (middlex_base + 50, self.y - self.window_height), (255, 255, 255), 2)
            cv2.rectangle(self.thrs, (rightx_base - 50, self.y), (rightx_base + 50, self.y - self.window_height), (255, 255, 255), 2)

            self.y -= self.window_height

        return self.thrs, self.lx, self.mx, self.rx

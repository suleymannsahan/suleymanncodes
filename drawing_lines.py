import cv2
import numpy as np
import sys

class DrawingLines:
    def __init__(self, line_image=None, slope=None, left_lines=None, right_lines=None,
                 avg_x1=None, avg_y1=None, avg_x2=None, avg_y2=None):
        self.line_image = line_image
        self.slope = slope
        self.left_lines = left_lines
        self.right_lines = right_lines
        self.avg_x1 = avg_x1
        self.avg_y1 = avg_y1
        self.avg_x2 = avg_x2
        self.avg_y2 = avg_y2
        
    def filling_line(self, mask, triangle):
        return cv2.fillPoly(mask, triangle, 255)
    
    def last_filling_image(self, canny_frame, mask):
        return cv2.bitwise_and(canny_frame, mask)
    
    def hough_lines(self, last_filling_image):
        return cv2.HoughLinesP(last_filling_image, rho=1, theta=np.pi/180, threshold=50, 
                               minLineLength=100, maxLineGap=50)

    def addWeighted(self, frame, line_image):
        return cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    
    def display_lines(self, frame, line_image, lines):
        self.line_image = np.zeros_like(frame)
        if lines is not None:
            self.left_lines = []
            self.right_lines = []
            for line in lines:
                for x1, y1, x2, y2 in line:
                    self.slope = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else 0  # Eğimi hesapla
                    if abs(self.slope) < 0.5:  # Çok yatay çizgileri filtrele
                        continue
                    if self.slope < 0:  # Sol şerit çizgileri
                        self.left_lines.append((x1, y1, x2, y2))
                    else:  # Sağ şerit çizgileri
                        self.right_lines.append((x1, y1, x2, y2))

            # Sol ve sağ çizgileri birleştirme ve çizim
            if self.left_lines:
                self.average_and_draw_lines(frame, line_image, self.left_lines)
            if self.right_lines:
                self.average_and_draw_lines(frame, line_image, self.right_lines)

        return self.line_image

    def average_and_draw_lines(self, frame, line_image, lines):
        # Tespit edilen çizgilerin ortalamasını alarak daha düzgün çizgi çiziyoruz
        x1s, y1s, x2s, y2s = zip(*lines)
        self.avg_x1, self.avg_y1 = int(np.mean(x1s)), int(np.mean(y1s))
        self.avg_x2, self.avg_y2 = int(np.mean(x2s)), int(np.mean(y2s))
        cv2.line(line_image, (self.avg_x1, self.avg_y1), (self.avg_x2, self.avg_y2), (255, 255, 255), 10)


import cv2
import numpy as np

video_path = "C:/Users/slmne/Downloads/circle_detection.mp4"
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (1025, 725))
    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([101, 50, 38])
    upper_blue = np.array([110, 255, 255])
    blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    
    lower_green = np.array([45, 100, 20])
    upper_green = np.array([75, 255, 255])
    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    
    lower_red = np.array([160, 20, 70])
    upper_red = np.array([190, 255, 255])
    red_mask = cv2.inRange(hsv_frame, lower_red, upper_red)

    blue_edges = cv2.Canny(blue_mask, 50, 100)
    green_edges = cv2.Canny(green_mask, 50, 100)
    red_edges = cv2.Canny(red_mask, 50, 100)

    
    blue_circles = cv2.HoughCircles(blue_edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=100)
    green_circles = cv2.HoughCircles(green_edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=100)
    red_circles = cv2.HoughCircles(red_edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=100)

    if blue_circles is not None:
        blue_circles = np.uint16(np.around(blue_circles))
        for circle in blue_circles[0, :]:
            x, y, r = circle
            cv2.circle(frame, (x, y), r, (255, 0, 0), 3)

    if green_circles is not None:
        green_circles = np.uint16(np.around(green_circles))
        for circle in green_circles[0, :]:
            x, y, r = circle
            cv2.circle(frame, (x, y), r, (0, 255, 0), 3)

    if red_circles is not None:
        red_circles = np.uint16(np.around(red_circles))
        for circle in red_circles[0, :]:
            x, y, r = circle
            cv2.circle(frame, (x, y), r, (0, 0, 255), 3)

    cv2.imshow("cap", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

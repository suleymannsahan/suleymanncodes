import cv2
import numpy as np

video_path = ("C:/Users/slmne/Downloads/line_video.mp4")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while(True):
    ret, frame = cap.read()
    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    sensitivity = 15
    lower_white = np.array([0, 0, 255-sensitivity], np.uint8)
    upper_white = np.array([255, sensitivity, 255], np.uint8)

    mask = cv2.inRange(hsv_frame, lower_white, upper_white)

    edges = cv2.Canny(mask, 50, 100)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=10, maxLineGap=100)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 7)


    cv2.imshow("video", frame)


    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


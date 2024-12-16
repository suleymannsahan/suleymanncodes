import cv2
import numpy as np
import sys
sys.path.append("/home/suleymann/Desktop/demo/suleyman_codes")
from filters_transform import FiltersTransform
from drawing_lines import DrawingLines
from sliding_window import HistogramSlidingWindow

video_path = "/home/suleymannsahan/Desktop/Lane.mp4"
cap = cv2.VideoCapture(video_path)

top_left = (537, 551)
bottom_left = (311, 679)
top_right = (695, 548)
bottom_right = (902, 681)

points1 = np.float32([top_left, bottom_left, top_right, bottom_right])
points2 = np.float32([[300, 0], [300, 480], [500, 0], [500, 480]])

filtering_color = FiltersTransform()

roi = FiltersTransform()
matrix = roi.getting_matrix(points1, points2)
dsize = (800, 480)

blur = FiltersTransform()
edges = FiltersTransform()
sobel = FiltersTransform()
drawing = DrawingLines()
appthreshold = FiltersTransform()
creating_histogram = HistogramSlidingWindow()
sliding = HistogramSlidingWindow()

def on_trackbar(val):
    pass

# Trackbar setup
def setup_trackbars(window_name):
    cv2.createTrackbar("Lower H", window_name, 0, 255, on_trackbar)
    cv2.createTrackbar("Lower L", window_name, 190, 255, on_trackbar)
    cv2.createTrackbar("Lower S", window_name, 0, 255, on_trackbar)
    cv2.createTrackbar("Upper H", window_name, 255, 255, on_trackbar)
    cv2.createTrackbar("Upper L", window_name, 255, 255, on_trackbar)
    cv2.createTrackbar("Upper S", window_name, 255, 255, on_trackbar)

window_name = "Trackbars"
cv2.namedWindow(window_name)
setup_trackbars(window_name)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    print(frame.shape)

    # Get trackbar positions
    lower_h = cv2.getTrackbarPos("Lower H", window_name)
    lower_l = cv2.getTrackbarPos("Lower L", window_name)
    lower_s = cv2.getTrackbarPos("Lower S", window_name)
    upper_h = cv2.getTrackbarPos("Upper H", window_name)
    upper_l = cv2.getTrackbarPos("Upper L", window_name)
    upper_s = cv2.getTrackbarPos("Upper S", window_name)

    # Update filter thresholds
    filtering_color.lower_white = (lower_h, lower_l, lower_s)
    filtering_color.upper_white = (upper_h, upper_l, upper_s)

    hls_frame = filtering_color.convert_to_hls(frame)
    white_mask = filtering_color.get_white_mask(hls_frame)
    
    trans_frame = roi.transform(frame, matrix, dsize)

    finding_contour = filtering_color.findContours(white_mask)

    transformed_frame = roi.transform(finding_contour, matrix, dsize)

    blurred_image = blur.applying_blur(finding_contour)
    
    canny_frame = edges.applying_canny(blurred_image)

    for point in points1:
        point = tuple(map(int, point))  
        cv2.circle(frame, point, 2, (125, 200, 255), -1)

    cv2.line(frame, top_left, bottom_left, (255, 0, 0), 1)
    cv2.line(frame, bottom_left, bottom_right, (255, 0, 0), 1)
    cv2.line(frame, bottom_right, top_right, (255, 0, 0), 1)
    cv2.line(frame, top_right, top_left, (255, 0, 0), 1)

    mask = np.zeros_like(canny_frame)
    triangle = np.array([[
    (0, mask.shape[0]),  # Sol alt köşe (yani görüntünün alt kısmı)
    (mask.shape[1] , mask.shape[0]),  # Sağ alt köşe
    (mask.shape[1] // 2, int(mask.shape[0] * 0.1))  # Orta üst, tam orta noktada
]])

    draw_the_line = drawing.filling_line(mask, triangle)

    last_filling_image = drawing.last_filling_image(canny_frame, mask)

    hough_lines_frame = drawing.hough_lines(last_filling_image)
    if hough_lines_frame is not None:
        print("lanes are detected")
    else:
        print("lanes are not detected")

    line_image = np.zeros_like(frame)
    addWeighted = drawing.addWeighted(frame, line_image)
    line_image = drawing.display_lines(frame, line_image, hough_lines_frame)

    sobel_combined = sobel.applying_sobel(white_mask)

    threshold = appthreshold.applying_threshold(transformed_frame)

    histogram, leftx_base, middlex_base, rightx_base = creating_histogram.applying_histogram(transformed_frame)

    thrs, lx, mx, rx = sliding.sliding_window(transformed_frame , leftx_base, middlex_base, rightx_base)

    lane_center = (leftx_base + rightx_base) // 2
    frame_center = frame.shape[1] // 2
    error = frame_center - lane_center

    cv2.imshow("Original Frame", frame)
    cv2.imshow("Transformed Frame", transformed_frame)
    #cv2.imshow("Threshold", threshold)
    cv2.imshow("Sliding Window", thrs)
    #cv2.imshow("sobel combined", sobel_combined)
    cv2.imshow("line frame", line_image)
    #cv2.imshow("canny frame", canny_frame)
    #cv2.imshow("ROI", mask)
    cv2.imshow("frameee", trans_frame)

    if cv2.waitKey(40) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

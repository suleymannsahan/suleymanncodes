import cv2


image_path = "/home/suleymann/Pictures/KGG.jpeg"

img = cv2.imread(image_path, 1)
resize_img = cv2.resize(img, (512,512))
blur_img = cv2.blur(resize_img, (7,7))

cv2.imshow("RESIZE", resize_img)
cv2.imshow("Blur Image", blur_img)
cv2.imwrite("BLUR.jpg", blur_img)

cv2.waitKey(0)
cv2.destroyAllWindows()


import cv2

image_path = "/home/suleymann/Pictures/KGG.jpeg"

img = cv2.imread(image_path, 1)
resize_img = cv2.resize(img, (512,512))


_, binary_image = cv2.threshold(resize_img,20,50,cv2.THRESH_BINARY)
boyutlandirma = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
erosion_img = cv2.erode(binary_image, boyutlandirma, iterations=1)


cv2.imshow("RESIZE IMAGE", resize_img)
cv2.imshow("BINARY IMAGE", binary_image)
cv2.imshow("EROSION IMAGE", erosion_img)
cv2.imwrite("EROSION.jpg", erosion_img)

cv2.waitKey(0)
cv2.destroyAllWindows()


import cv2

i = cv2.imread('original.jpg')
print(i.shape)
i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
print(i.shape)
i = 255-i
print(i.shape)
cv2.imshow('gray', i)
cv2.waitKey()
cv2.destroyAllWindows()

import cv2

n=2020
p=r'data/'

path=p+'one/one_'
print(path)
for i in range (0, n):
	newpath=path+str(i)+'.png'
	img=cv2.imread(newpath)
	img=cv2.resize(img, (100, 120))
	# cv2.imshow('ONE', img)
	cv2.imwrite(newpath, img)
	print(newpath)

path=p+'two/two_'
print(path)
for i in range (0, n):
	newpath=path+str(i)+'.png'
	img=cv2.imread(newpath)
	img=cv2.resize(img, (100, 120))
	# cv2.imshow('TWO', img)
	cv2.imwrite(newpath, img)
	print(newpath)

path=p+'three/three_'
print(path)
for i in range (0, n):
	newpath=path+str(i)+'.png'
	img=cv2.imread(newpath)
	img=cv2.resize(img, (100, 120))
	# cv2.imshow('THREE', img)
	cv2.imwrite(newpath, img)
	print(newpath)

path=p+'four/four_'
print(path)
for i in range (0, n):
	newpath=path+str(i)+'.png'
	img=cv2.imread(newpath)
	img=cv2.resize(img, (100, 120))
	cv2.imwrite(newpath, img)
	# cv2.imshow('FOUR', img)
	print(newpath)

path=p+'five/five_'
print(path)
for i in range (0, n):
	newpath=path+str(i)+'.png'
	img=cv2.imread(newpath)
	img=cv2.resize(img, (100, 120))
	# cv2.imshow('FIVE', img)
	cv2.imwrite(newpath, img)
	print(newpath)


path=p+'blank/blank_'
print(path)
for i in range (0, n):
	newpath=path+str(i)+'.png'
	img=cv2.imread(newpath)
	img=cv2.resize(img, (100, 120))
	# cv2.imshow('BLANK', img)
	cv2.imwrite(newpath, img)
	print(newpath)

# cv2.destroyAllWindows()
import cv2
import numpy as np

from utils import resize

def preprocess_image(img_path):
	img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	img = _scan(img)
	img = resize(img, 2048, 3736)
	cv2.imwrite('aligned.png', img)
	img = _binarize(img)
	cv2.imwrite('binary-aligned.png', img)
	img = _denoise(img)
	cv2.imwrite('denoised.png', img)

	return img


def _order_points(pts):
    '''Rearrange coordinates to order:
      top-left, top-right, bottom-right, bottom-left'''
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    # Top-left point will have the smallest sum.
    rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
    rect[2] = pts[np.argmax(s)]
 
    diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
    rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect.astype('int').tolist()

def _find_dest(pts):
    (tl, tr, br, bl) = pts
    # Finding the maximum width.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    # Finding the maximum height.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # Final destination co-ordinates.
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
 
    return _order_points(destination_corners)

'''
def pad(corners, padding):
	x1, y1 = corners[0]
	x2, y2 = corners[1]
	x3, y3 = corners[2]
	x4, y4 = corners[3]

	xc, yc = ((x1+x2+x3+x4)/4, (y1+y2+y3+y4)/4) # centroid

	cp1 = ((x1-xc)*padding, (y1-yc)*padding)
	cp2 = ((x2-xc)*padding, (y2-yc)*padding)
	cp3 = ((x3-xc)*padding, (y3-yc)*padding)
	cp4 = ((x4-xc)*padding, (y4-yc)*padding)

	x1 = xc + cp1[0]
	y1 = yc + cp1[1]
	x2 = xc + cp2[0]
	y2 = yc + cp2[1]
	x3 = xc + cp3[0]
	y3 = yc + cp3[1]
	x4 = xc + cp4[0]
	y4 = yc + cp4[1]

	new_corners = [[x1,y1], [x2, y2], [x3,y3], [x4,y4]]

	return new_corners
'''
# expands corners horizontally to account for the small width between ArUco markers and the scoresheet's edges
def _pad(corners, padding):
	x1, y1 = corners[0]
	x2, y2 = corners[1]
	x3, y3 = corners[2]
	x4, y4 = corners[3]

	dx_12 = (x2 - x1) * padding
	dy_12 = (y2 - y1) * padding

	dx_43 = (x3 - x4) * padding
	dy_43 = (y3 - y4) * padding

	x1 -= dx_12
	y1 -= dy_12

	x2 += dx_12
	y2 += dy_12

	x3 += dx_43
	y3 += dy_43

	x4 -= dx_43
	y4 -= dy_43

	new_corners = [[x1,y1], [x2, y2], [x3,y3], [x4,y4]]

	return new_corners

def _scan(img):

	# Create a copy of resized original image for later use
	resized_img = img.copy()

	# Resize image to workable size
	dim_limit = 1080
	max_dim = max(img.shape)
	if max_dim > dim_limit:
		resize_scale = dim_limit / max_dim
		resized_img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)

	dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
	parameters =  cv2.aruco.DetectorParameters()
	detector = cv2.aruco.ArucoDetector(dictionary, parameters)
	(corners, ids, rejected) = detector.detectMarkers(resized_img)

	img_corners = []

	# verify that *exactly* 4 ArUCo markers were found
	if [[i] in ids for i in range(4)]:
		# flatten the ArUco IDs list
		ids = ids.flatten()
		# loop over the detected ArUCo corners
		for (markerCorner, markerID) in zip(corners, ids):
			# extract the marker corners (which are always returned in
			# top-left, top-right, bottom-right, and bottom-left order)
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners

			if markerID == 0:
				img_corners.append(topLeft)
			if markerID == 1:
				img_corners.append(topRight)
			if markerID == 2:
				img_corners.append(bottomLeft)
			if markerID == 3:
				img_corners.append(bottomRight)
	else:
		raise Exception("Incorrect number of ArUco markers detected.")

	# For 4 corner points being detected.
	img_corners = _order_points(img_corners)

	# Horizontally pad the corners
	img_corners = _pad(img_corners, 1/65)

	img_corners = [[corner[0]/resize_scale, corner[1]/resize_scale] for corner in img_corners]
	destination_corners = _find_dest(img_corners)


	h, w = img.shape[:2]
	# Getting the homography.
	M = cv2.getPerspectiveTransform(np.float32(img_corners), np.float32(destination_corners))
	# Perspective transform using homography.
	final = cv2.warpPerspective(img, M, (destination_corners[2][0], destination_corners[2][1]),
								flags=cv2.INTER_LINEAR)

	return final

def _binarize(img):
	# do adaptive threshold on gray image
	binarized_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 15)

	# standard practice is white foreground / black background
	binarized_img = 255 - binarized_img
	cv2.imwrite('binary-aligned.png', binarized_img)

	return binarized_img

def _denoise(img):
	# Start by finding all of the connected components (white blobs in your image).
	# 'im' needs to be grayscale and 8bit.
	#img_uint8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img_uint8 = img.astype(np.uint8)

	num_components, img_with_separated_components, stats, _ = cv2.connectedComponentsWithStats(img_uint8)
	# im_with_separated_blobs is an image where each detected blob has a different pixel value ranging from 1 to nb_blobs - 1.
	# The background pixels have value 0.
	# 'stats' (and the silenced output 'centroids') provides information about the blobs. See the docs for more information. 
	# Here, we're interested only in the size of the blobs :
	sizes = stats[:, cv2.CC_STAT_AREA]
	# You can also directly index the column with '-1' instead of 'cv2.CC_STAT_AREA' as it's the last column.

	# A small gotcha is that the background is considered as a blob, and so its stats are included in the stats vector at position 0.

	# minimum size of particles we want to keep (number of pixels).
	# here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever.
	min_size = 150  

	# create empty output image with will contain only the biggest components
	img = np.zeros_like(img)

	# for every component in the image, keep it only if it's above min_size.
	# we start at 1 to avoid considering the background
	for component in range(1, num_components):
		if sizes[component] >= min_size:
			img[img_with_separated_components == component] = 255

	# now that the image is denoised, we can close holes with MORPH_CLOSE
	kernel = np.ones((2, 1), np.uint8)
	img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)

	return img
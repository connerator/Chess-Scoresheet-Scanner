
import cv2
import numpy as np

def extract_move_boxes(img):
    # (roi, x, y)
    boxes_and_coords = _find_boxes(img)
    cv2.imwrite('box.png', boxes_and_coords[78][0])

    boxes_and_coords = _sort_boxes(boxes_and_coords)
    i = 0
    for box in boxes_and_coords:
        roi, x, y = box
        h, w = roi.shape
        cv2.putText(img,str(i),
        (x, y+h), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1,
        127,
        1,
        2)
        i += 1

        cv2.imwrite("denoised.png", img)

    move_boxes = [x[0] for x in boxes_and_coords]

    cv2.imwrite('box.png', move_boxes[0])

    move_boxes = [_remove_borders(box) for box in move_boxes]

    #cv2.imwrite('box.png', move_boxes[78])

    return move_boxes
'''
def _find_boxes(img):
    # invert image because move boxes are negative space
    inverted_img = 255 - img
    # connected component search
    num_components, img_with_separate_components, stats, _ = cv2.connectedComponentsWithStats(inverted_img)

    move_boxes = []
    i = 0
    # first component is always the background (for some reason), so we start at 1
    for component in range(1, num_components):
        x = stats[component, cv2.CC_STAT_LEFT]
        y = stats[component, cv2.CC_STAT_TOP]
        w = stats[component, cv2.CC_STAT_WIDTH]
        h = stats[component, cv2.CC_STAT_HEIGHT]
        
        # only move boxes have this width to height ratio
        if 2.5 < w/h < 4.5 and 54000 < w*h < 63000:

            mask = np.zeros_like(img)
            mask[img_with_separate_components == component] = 255

            roi = mask[y:y+h, x:x+h]


            move_boxes.append((component, x, y, w, h))
            print(w*h)

            cv2.putText(img,str(i), 
            (x, y+h), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1,
            127,
            1,
            2)
            i += 1

            cv2.imwrite("denoised.png", img)

        
    return move_boxes
'''
def _find_boxes(img):
    # invert image because move boxes are negative space
    inverted_img = 255 - img

    move_boxes = []
    i = 0

    contours, _ = cv2.findContours(inverted_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over each contour and save it as a new image
    for _, contour in enumerate(contours):

        # Create a mask for the current contour
        mask = np.zeros_like(inverted_img)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Extract the contour region from the original image using the mask
        contour_region = cv2.bitwise_and(inverted_img, inverted_img, mask=mask)
        
        # Find the bounding box of the contour to crop the region
        x, y, w, h = cv2.boundingRect(contour)
        
        if 2.5 < w/h < 4.5 and 54000 < w*h < 63000:

            roi = contour_region[y:y+h, x:x+w]

            # un-invert image
            roi = 255 - roi

            move_boxes.append((roi, x, y))

    return move_boxes
        

# sort boxes into move-order
def _sort_boxes(move_boxes):

    # sort by x, which puts each set of 20 vertical columns in place
    move_boxes.sort(key=lambda x: x[1])

    # now sort each vertical column by y
    move_boxes[:20] = sorted(move_boxes[:20], key=lambda x: x[2])
    move_boxes[20:40] = sorted(move_boxes[20:40], key=lambda x: x[2])
    move_boxes[40:60] = sorted(move_boxes[40:60], key=lambda x: x[2])
    move_boxes[60:80] = sorted(move_boxes[60:80], key=lambda x: x[2])

    # alternate between rows 1 and 2, and then 3 and 4
    # this sorts the boxes into move-order
    temp = []
    for i in range(20):
        temp.append(move_boxes[i])
        temp.append(move_boxes[i+20])
    for i in range(40, 60):
        temp.append(move_boxes[i])
        temp.append(move_boxes[i+20])
    move_boxes = temp

    return move_boxes

'''
def _find_ROIs(img, move_boxes):
    rois = []
    for move_box in move_boxes:
        _, x, y, w, h, = move_box

        # crop out the border
        crop = 5
        roi = img[y+crop:y+h-crop, x+crop:x+w-crop]

        rois.append(roi)
    
    return rois
'''

def _remove_borders(move_box):
    y, x = move_box.shape

    for i in range(x):
        if move_box[0][i] == 255:
            cv2.floodFill(move_box, None, (i, 0), 0)
        if move_box[y-1][i] == 255:
            cv2.floodFill(move_box, None, (i, y-1), 0)

    for i in range(y):
        if move_box[i, 0] == 255:
            cv2.floodFill(move_box, None, (0, i), 0)
        if move_box[i, x-1] == 255:
            cv2.floodFill(move_box, None, (x-1, i), 0)

    return move_box
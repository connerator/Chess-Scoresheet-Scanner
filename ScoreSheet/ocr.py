import cv2
import numpy as np
import tensorflow as tf

import itertools

class OCRModel:
    def __init__(self, weights_path):
        """Initialize the OCR model with the given weights."""
        
        self.model = tf.keras.Sequential([
            tf.keras.Input((28, 28, 1)),
            tf.keras.layers.Conv2D(32,3),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512,activation='relu'),
            tf.keras.layers.Dense(128,activation='relu'),
            tf.keras.layers.Dense(47,activation='softmax')
        ])


        self.model.load_weights(weights_path)

        self.index_to_char = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 
                         10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I',
                         19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R',
                         28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a',
                         37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r',
                         46: 't'}
        
        self.valid_chars = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 20, 23, 24, 26, 27, 33, 36, 37, 38, 39, 40, 41, 42, 46] # 46 (t) to recognize +
        self.valid_to_true_char = {'0':'O', 'C':'c', '9':'g', 'X':'x', 't':'+'}

    # returns top n predictions in a tuple
    def perform_ocr(self, move_box, n=3):
        char_boxes = self._find_chars(move_box)

        if len(char_boxes) == 0:
            return tuple()

        # gaussian blur -> inter-cubic rescale to 28x28 with padding -> convert to (1, 28, 28, 1) tensor
        char_boxes = [self._format_to_mnist(box) for box in char_boxes]

        potential_char_matrix = []
        potential_moves = []

        for box in char_boxes:
            model_outputs = list(self.model(box).numpy())[0]
            #print(model_outputs)
            #print()
            # list in the following format: (probability, character), ie (0.94, 'K')
            potential_chars = []
            for i, probability in enumerate(model_outputs):

                # exlude non-valid characters
                if i not in self.valid_chars:
                    continue

                character = self.index_to_char[i]
                if character in self.valid_to_true_char:
                    character = self.valid_to_true_char[character]

                potential_chars.append((probability, character))

            # sort by probability
            potential_chars.sort(reverse=True)

            # truncate to only first 5, to keep computations manageable
            potential_chars = potential_chars[:10]

            potential_char_matrix.append(potential_chars)

        # cartesian product across axis 0 of the matrix
        potential_moves = list(itertools.product(*potential_char_matrix))
        temp = []
        # char_tuple example: ((1.0, 'Q'), (0.99066085, 'f'), (0.99999964, '3'), (1.0, '+'))
        for char_tuple in potential_moves:
            total_probability = 1
            move = ""
            for probability, character in char_tuple:
                total_probability *= probability
                move += character
            temp.append((total_probability, move))
        
        # temporary fix for mis-identifying 'e' as 'Q'
        for probability, move in temp:
            if 'Q' in move:
                temp.append((probability-0.01, move.replace('Q', 'e')))
            if move == '0+0' or move == 'O+O':
                temp.append((probability+0.01, move.replace('+', '-')))

        temp.sort(reverse=True)
        potential_moves = temp            
        
        return tuple(potential_moves[:n])
    
    def _find_chars(self, move_box):
        # Find the contours (RETR_EXTERNAL makes sure we only return the outermost contours)
        contours, _ = cv2.findContours(move_box, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # sort by x
        contours = sorted(contours, key=lambda x:cv2.boundingRect(x)[0])

        char_ROIs = []

        # Loop over each contour and save it as a new image
        for _, contour in enumerate(contours):

            # Create a mask for the current contour
            mask = np.zeros_like(move_box)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            # Extract the contour region from the original image using the mask
            contour_region = cv2.bitwise_and(move_box, move_box, mask=mask)
            
            # Find the bounding box of the contour to crop the region
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out border pixels / noise

            if w*h <= 200 or (24 * min(w, h)) // max(w, h) == 0:
                continue

            cropped_contour = contour_region[y:y+h, x:x+w]

            char_ROIs.append(cropped_contour)
        
        return char_ROIs
    
    def _format_to_mnist(self, img):

        img = cv2.GaussianBlur(img,(5,5),1)

        img_h, img_w = img.shape
        dim_size_max = max(img.shape)

        if dim_size_max == img_w:
            im_h = (24 * img_h) // img_w
            if im_h <= 0 or img_w <= 0:
                print("Invalid Image Dimention: ", im_h, img_w, img_h)
            tmp_img = cv2.resize(img, (24,im_h),0,0,cv2.INTER_CUBIC)
        else:
            im_w = (24 * img_w) // img_h
            if im_w <= 0 or img_h <= 0:
                print("Invalid Image Dimention: ", im_w, img_w, img_h)
            tmp_img = cv2.resize(img, (im_w, 24),0,0,cv2.INTER_CUBIC)

        out_img = np.zeros((28, 28), dtype=np.ubyte)

        nb_h, nb_w = out_img.shape
        na_h, na_w = tmp_img.shape
        y_min = (nb_w) // 2 - (na_w // 2)
        y_max = y_min + na_w
        x_min = (nb_h) // 2 - (na_h // 2)
        x_max = x_min + na_h

        out_img[x_min:x_max, y_min:y_max] = tmp_img

        # normalize
        out_img = out_img / 255.

        # training images are vertically flipped and 90 degrees clockwise, so we replicate this here
        out_img = np.rot90(out_img,1,(0,1))
        out_img = np.flip(out_img, axis=0)

        out_img = out_img.reshape(1, 28, 28, 1)

        return out_img
    




# start: piece OR file
# piece -> 
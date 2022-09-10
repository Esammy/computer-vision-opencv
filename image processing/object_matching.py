import sys
import cv2
import numpy as np

class objMatching:
    def __init__(self, image):
        self.image=image

    def show(self, image, caption = 'Image'):
        cv2.imshow(caption, image)
        cv2.waitKey(0)

    def draw_matches(img1, keypoints1, img2, keypoints2, matches):
        rows1, cols1 = img1.shape[:2]
        rows2, cols2 = img2.shape[:2]
        # Create a new output image that concatenates the two images
        together
        output_img = np.zeros((max([rows1,rows2]), cols1+cols2, 3),
        dtype='uint8')
        output_img[:rows1, :cols1, :] = np.dstack([img1, img1, img1])
        output_img[:rows2, cols1:cols1+cols2, :] = np.dstack([img2, img2,
        img2])
        # Draw connecting lines between matching keypoints
        for match in matches:
            # Get the matching keypoints for each of the images
            img1_idx = match.queryIdx
            img2_idx = match.trainIdx
            (x1, y1) = keypoints1[img1_idx].pt
            (x2, y2) = keypoints2[img2_idx].pt
            # Draw a small circle at both co-ordinates and then draw a
            line
            radius = 4
            colour = (0,255,0) # green
            thickness = 1
            cv2.circle(output_img, (int(x1),int(y1)), radius, colour,
            thickness)
            cv2.circle(output_img, (int(x2)+cols1,int(y2)), radius,
            colour, thickness)
            cv2.line(output_img, (int(x1),int(y1)),
            (int(x2)+cols1,int(y2)), colour, thickness)
        return output_img

    def match(self, img1, img2):
        # Initialize ORB detector
        orb = cv2.ORB()

        # Extract keypoints and descriptors
        keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
        keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

        # Create Brute Force matcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        # Match descriptors
        matches = bf.match(descriptors1, descriptors2)
        # Sort them in the order of their distance
        matches = sorted(matches, key = lambda x:x.distance)
        # Draw first 'n' matches
        img3 = draw_matches(img1, keypoints1, img2, keypoints2, matches[:30])
        return img3

if __name__=='__main__':
    image = 'Eleojo.jpg'
    img1 = cv2.imread(image, 0) # query image (rotatedsubregion)
    img2 = cv2.imread(image, 0)# train image (full image)

    test = objMatching(image)
    matched_img = test.match(img1, img2)
    test.show(matched_img)    


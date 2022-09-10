import numpy as np
import cv2

class ImageFiltering:
    def __init__(self, image):
        self.image = image
        self.path = 'saved_filtered_img.jpg'

    def readImage(self):
        img = cv2.imread(self.image)
        rows, cols = img.shape[:2]
        return img, rows, cols

    def vignetteFilter(self, shape_para = 1.5, color_range = 255, brightness = 200):
        img, rows, cols = self.readImage()

        # Generating vignette mask using Gaussian Kernels
        kernel_x = cv2.getGaussianKernel(int(shape_para*cols), int(brightness))
        kernel_y = cv2.getGaussianKernel(int(shape_para*rows), int(brightness))
        kernel = kernel_y * kernel_x.T
        mask = color_range * kernel / np.linalg.norm(kernel)
        mask = mask[int((shape_para-1)*rows):, int((shape_para-1)*cols):]
        
        output = np.copy(img)

        # Applying the mask to each channel in the input image
        for i in range(3):
            output[:,:,i] = output[:,:,i] * mask

        return output

    def show(self, image, caption = 'Image'):
        cv2.imshow(caption, image)
        cv2.waitKey(0)

    def save(self, path, image_frame):
        cv2.imwrite(path, image_frame)
        return str(path)

    def imageBlurring(self, m = 1):
        img, rows, cols = self.readImage()
        

        try:
            normalization = m * m # Normalizing the matrix

            kernel_identity = np.array([[0,0,0], [0,1,0], [0,0,0]])
            kernel_mxm = np.ones((m,m), np.float32) / normalization

            idnty_output = cv2.filter2D(img, -1, kernel_identity) 
            mxm_output = cv2.filter2D(img, -1, kernel_mxm)
            
            return idnty_output, mxm_output
        except:
            print(" Please enter another number")
            return self.imageBlurring()
        
    def edgeDetection(self, lower_threshod=50, upper_threshold=240):
        img, rows, cols = self.readImage()

        canny = cv2.Canny(img, lower_threshod, upper_threshold)
        return canny

    def motionBlur(self, size = 15):
        img, rows, cols = self.readImage()

        # Generating the kernel
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size-1)/2),:] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size

        # Applying the kernel to the input image
        output = cv2.filter2D(img, -1, kernel_motion_blur)
        return output

    def sharpening(self):
        img, rows, cols = self.readImage()

        # Generating the kernel
        kernel_sharpen_default = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        kernel_sharpen_finetune = np.array([[-1,-1,-1,-1,-1],
                                           [-1,2,2,2,-1],
                                           [-1,2,8,2,-1],
                                           [-1,2,2,2,-1],
                                           [-1,-1,-1,-1,-1]]) /8

        # Applying different kernels to the input image
        output_1 = cv2.filter2D(img, -1, kernel_sharpen_default)
        output_2 = cv2.filter2D(img, -1, kernel_sharpen_finetune)
        return output_1, output_2

    def contrast(self):
        img, rows, cols = self.readImage()

        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        # Equalize the histogram of the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

        # Convert the YUV image back to RGB format
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output


    def cartoonize_image(self, ds_factor=4, sketch_mode=False):
        # Read image
        img, rows, cols = self.readImage()

        # Convert image to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply median filter to the grayscale image
        img_gray = cv2.medianBlur(img_gray, 7)

        # Detect edges in the image and threshold it
        edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=5)
        ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)

        # 'mask' is the sketch of the image
        if sketch_mode:
            return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Resize the image to a smaller size for faster computation
        img_small = cv2.resize(img,
                               None,
                               fx=1.0/ds_factor,
                               fy=1.0/ds_factor,
                               interpolation=cv2.INTER_AREA)
        num_repetitions = 10
        sigma_color = 5
        sigma_space = 7
        size = 5

        # Apply bilateral filter the image multiple times
        for i in range(num_repetitions):
            img_small = cv2.bilateralFilter(img_small,
                                        size,
                                        sigma_color,
                                        sigma_space)
        img_output = cv2.resize(img_small,
                                None,
                                fx=ds_factor,
                                fy=ds_factor,
                                interpolation=cv2.INTER_LINEAR)
        dst = np.zeros(img_gray.shape)

        # Add the thick boundary lines to the image using 'AND' operator
        dst = cv2.bitwise_and(img_output, img_output, mask=mask)
        return dst

if __name__ == '__main__':
    image = 'Eleojo.jpg'
    test = ImageFiltering(image)
    vig_img = test.vignetteFilter(1.6, 255, 10000)
    idnty_, mxm = test.imageBlurring(2)
    edgeD = test.edgeDetection()
    motionBlur = test.motionBlur(10)
    shrp_1, shrp_2 = test.sharpening()
    contrast = test.contrast()
    catoon = test.cartoonize_image(sketch_mode=False)# if False it is catoon, if true it sketch
    test.show(vig_img, caption = 'Filtered Image')

    # Original image
    img, rows, cols =test.readImage()
    #test.show(img, caption = 'Original Image')


''' Note: (1st = it starts shading the image from the top
    right corer down bottom to bottom left and to the top left.
    starting from 1.0, 1.3, 1.5, 1.6, 1.7, 1.8, 2.0, 2.2, 2.4, 2.5
    2nd = it shows the color intensity of the image,
    3rd = it darken and brightens the shaded area by the '1st' param. when
    the value is extremly large it normalizes the image to a uniform shade)
'''

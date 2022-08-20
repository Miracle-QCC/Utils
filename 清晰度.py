import cv2
import numpy as np


def get_picture_sharpness(image_path):
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_var = str(int(cv2.Laplacian(image_gray, cv2.CV_64F).var()))
    # image_var = str(int(cv2.Laplacian(image_gray, cv2.CV_16U).var()))

    font_face = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 1
    thickness = 1
    baseline = 0

    var_size = cv2.getTextSize(image_var, font_face, font_scale, thickness)
    # 清晰度值的绘制位置
    draw_start_point = (20, var_size[0][1] + 10)
    cv2.putText(image, image_var, draw_start_point, font_face, font_scale, (0, 0, 255), thickness)

    cv2.imshow('frame', image)
    cv2.waitKey(0)

    return image_var


def fun_blur(img):
    # Laplace model
    # img =cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    s = cv2.Laplacian(img,cv2.CV_8U)
    return np.abs(s).mean()

def fun_blur8(img):
    # img =cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    fx = np.power(img[:,1:-1]*2 - img[:,:-2] - img[:,2:],2)
    fy = np.power(img[1:-1,:]*2 - img[:-2,:] - img[:-2,:],2)
    return np.sqrt(fx.mean()+fy.mean())

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


def get_std(image):
    mean = np.mean(image)
    m, n = image.shape[0], image.shape[1]
    # std = np.power(image - mean,2)
    # std = np.sum(std)
    # std = np.sqrt(std) / (m * n)
    std = np.sqrt(np.sum(np.power(image - mean, 2))) / (m * n)
    return std


if __name__ == '__main__':

    image = cv2.imread("origin.ppm",0)
    # es = fun_blur(image)
    # print("Edge Sharpness:",es)
    get_std(image)

    # the mean of std
    
# if fm < 100:
#     text = "Blurry"
# # show the image
# cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
# cv2.imshow("Image", image)


# print(get_picture_sharpness("origin.ppm"))
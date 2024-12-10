import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, thin
from skimage.io import imshow
face_cascade = cv2.CascadeClassifier('/home/zizo/Downloads/haarcascade_frontalface_default.xml')

def RGBtoYCbCr (R, G, B):
    R = int(R)
    G = int(G)
    B = int(B)
    R /= 255.0
    G /= 255.0
    B /= 255.0
    Y = 16 + (65.481 * R + 128.553 * G + 24.966 * B)
    Cb = 128 + (-37.797 * R - 74.203 * G + 112.0 * B)
    Cr = 128 + (112.0 * R - 93.786 * G - 18.214 * B)
    return Y, Cb, Cr

def train(path):    
    content = ""
    with open(path, 'r') as file:
        content = file.read()
    entries = content.split('\n')
    dataset = dict()
    for line in entries:
        if line:
            R, G, B, label = line.split()
            label = int(label)
            if(label not in dataset):
                dataset[label] = []
            Y, Cb, Cr = RGBtoYCbCr(R, G, B)
            dataset[label].append([Cb, Cr])
    return dataset

def get_mean_cov(dataset):
    mean = dict()
    cov = dict()
    for label in dataset:
        data = np.array(dataset[label])
        mean[label] = np.mean(data, axis=0)
        cov[label] = np.cov(data, rowvar=False)
    return mean, cov

def prob_c_label(C, mean, cov):

    C = np.array(C)
    mean = np.array(mean)
    cov = np.array(cov)

    C_diff = C - mean
    inv_cov = np.linalg.inv(cov)
    prob = np.exp(-0.5 * np.sum(C_diff @ inv_cov * C_diff, axis=-1))

    norm_factor = np.sqrt(np.linalg.det(cov) * (2 * np.pi) ** C.shape[1])
    
    return prob / norm_factor

def prob_skin_c(C, skinMean, skinCov, nonSkinMean, nonSkinCov):
    probCskin = prob_c_label(C, skinMean, skinCov)
    probCnonSkin = prob_c_label(C, nonSkinMean, nonSkinCov)

    return probCskin / (probCskin + probCnonSkin)

def cleanMask(skin_mask):
    binary_mask = cv2.threshold(skin_mask, 0.15, 1, cv2.THRESH_BINARY)[1]
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=2)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, np.ones((3, 3)), iterations=2)
    return binary_mask

def show_images(images,titles=None):
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()

def do_skeltonization(to_skeletonize):
    return cv2.ximgproc.thinning(cv2.GaussianBlur((to_skeletonize * 255).astype(np.uint8), (5,5), 0)) / 255

def get_end_points(skeletonized):
    coordinates = []
    for i in range(1, skeletonized.shape[0] - 1):
        for j in range(1, skeletonized.shape[1] - 1):
            pix = skeletonized[i, j]
            if pix == 0:
                continue
            count = 0
            for y in range(-1, 2):
                for x in range(-1, 2):
                    pix = skeletonized[i + y, j + x]
                    if pix != 0:
                        count += 1
            if count == 2:
                coordinates.append((i, j))
    return coordinates

def draw_circles(img, coordinates):
    res_5 = np.copy(img)
    for (x, y) in coordinates:
        cv2.circle(res_5, (y, x), radius=5, color=(0, 0, 255), thickness=-1)  
    return res_5

def main():
    cap = cv2.VideoCapture(0)

    path = './Skin_NonSkin.txt'
    dataset = train(path)
    mean, cov = get_mean_cov(dataset)
    skin_mean = mean[1]
    skin_cov = cov[1]
    non_skin_mean = mean[2]
    non_skin_cov = cov[2]

    while True:

        #image = cv2.imread("/home/zizo/CMP Year three/Semester 1/Image Processing/Project/test_images/Palm_fingers.jpg")
        ret, frame = cap.read()
        if not ret:
            break

        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (256, 256))
        
        original = frame.copy()
        YCC = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb)
        
        frame = frame.astype(np.float64) / 255
        C = YCC[:, :, 1:]
        skin_mask = np.zeros((frame.shape[0], frame.shape[1]))
        skin_mask = prob_skin_c(C, skin_mean, skin_cov, non_skin_mean, non_skin_cov)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        skin_mask_dilated = cv2.dilate(skin_mask, kernel)
        

        skeletonized = do_skeltonization(np.copy(skin_mask_dilated))
        
        coordinates = get_end_points(skeletonized)
        result = draw_circles(frame, coordinates=coordinates)
        cv2.imshow('Result', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
from tkinter import *
import numpy as np
from numpy import linalg as la
import cv2
import matplotlib as plt
from collections import Counter
import boto3

#
# session = boto3.Session(
#     aws_access_key_id = '',
#
# )

caleDS = r'C:/Users/Sebi/Desktop/fete/'

nrPers = 40
nrPozePers = 8
rezolutie = 112 * 92


def training_matrix(caleDS):

    A = np.zeros((rezolutie, nrPers * nrPozePers))

    for nr in range(1, nrPers + 1):
        for im in range(1, nrPozePers + 1):

            img = cv2.imread(f'{caleDS}/s{nr}/{im}.pgm', 0)
            img = np.array(img)

            imgVect = np.reshape(img, (-1,))
            A[:, (nr - 1) * nrPozePers + im - 1] = imgVect

    print('Shape ul matricii de antrenare:', A.shape)
    return A


def NN(matrice, img, norma):

    z = np.zeros((nrPozePers * nrPers))

    for i in range(len(z)):
        z[i] = np.linalg.norm(img - matrice[:, i], norma)

    poz = np.argmin(z)
    return poz

def kNN(matrice, img, norma, k):
    z = np.zeros((nrPozePers * nrPers))

    for i in range(len(z)):
        z[i] = np.linalg.norm(img - matrice[:, i], norma)

    nearest_indices = np.argsort(z)[:k]
    nearest_classes = [(idx // nrPozePers) + 1 for idx in nearest_indices]

    most_common = Counter(nearest_classes).most_common(1)[0][0]
    return most_common

#

matrice = training_matrix(caleDS)

cale_poza_test = f'{caleDS}/s18/10.pgm'
test_img = cv2.imread(cale_poza_test, 0)
test_img = np.array(test_img).reshape(-1)

## NN for recogn
norma = 2

predicted_index = NN(matrice, test_img, norma)

predicted_image_path = f'{caleDS}/s{int(predicted_index / nrPozePers + 1)}/{predicted_index % nrPozePers + 1}.pgm'

predicted_image = cv2.imread(predicted_image_path, 0)
cv2.imshow('Predicted Image', predicted_image)
cv2.imshow('Test Image', cv2.imread(cale_poza_test, 0))

#kNN for recogn
k = 7
predicted_class = kNN(matrice, test_img, norma, k)
print(f'The predicted class using kNN  for the test is: Person {predicted_class}')

cv2.waitKey()
cv2.destroyAllWindows()

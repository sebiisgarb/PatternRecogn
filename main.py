import time
from tkinter import *
import numpy as np
from numpy import linalg as la
import cv2
from collections import Counter
from sklearn.model_selection import KFold

caleDS = r'C:\Users\Popescu Sebastian\Desktop\PatternRecogn\att_faces'

nrPers = 40
nrPozePers = 7

nrPozePersTotal = 10

rezolutie = 112 * 92


def load_data(caleDS):
    """
    Funcție pentru a încărca toate imaginile într-un dicționar.
    """
    data = []
    etichete = []

    # Iterăm prin fiecare persoană și fiecare poză
    for nr in range(1, nrPers + 1):
        for im in range(1, nrPozePersTotal + 1):
            img = cv2.imread(f'{caleDS}/s{nr}/{im}.pgm', 0)
            if img is not None:
                img = np.array(img).reshape(-1)  # Convertim imaginea într-un vector
                data.append(img)
                etichete.append(nr)  # Fiecare imagine primește ca etichetă numărul persoanei

    return np.array(data), np.array(etichete)

# def statistici(alg, matrice_antrenare, imagini_test, etichete_test, norma, k=None):
#     nr_recunoasteri_corecte = 0
#     timpi_executie = []
#
#     for i, imagine_test in enumerate(imagini_test):
#         start_time = time.perf_counter()
#
#         if k is None:
#             predicted_index = alg(matrice_antrenare, imagine_test, norma)
#             predicted_class = etichete_test[predicted_index]
#         else:
#             predicted_class = alg(matrice_antrenare, imagine_test, norma, k)
#
#         end_time = time.perf_counter()
#         exec_time = end_time - start_time
#         timpi_executie.append(exec_time)
#
#         # Verificăm dacă recunoașterea este corectă
#         if predicted_class == etichete_test[i]:
#             nr_recunoasteri_corecte += 1
#
#         # Calculăm rata de recunoaștere
#     rata_recunoastere = nr_recunoasteri_corecte / len(imagini_test)
#
#     # Calculăm timpul mediu de interogare
#     timp_mediu = np.mean(timpi_executie)
#
#     return rata_recunoastere, timp_mediu

def training_matrix(caleDS):
    A = np.zeros((rezolutie, nrPers * nrPozePersTotal))

    for nr in range(1, nrPers + 1):
        for im in range(1, nrPozePersTotal + 1):
            img = cv2.imread(f'{caleDS}/s{nr}/{im}.pgm', 0)
            img = np.array(img)

            imgVect = np.reshape(img, (-1,))
            A[:, (nr - 1) * nrPozePersTotal + im - 1] = imgVect

    print('Dimensiunea matricii de antrenare:', A.shape)
    return A


def NN(matrice, img, norma):

    img = img.flatten()
    # Vector pentru a stoca distanțele dintre imaginea de test și fiecare imagine din matrice
    z = np.zeros((matrice.shape[1]))

    # Calculăm norma (distanța) între imaginea de test și fiecare imagine din matrice
    for i in range(len(z)):
        z[i] = np.linalg.norm(img - matrice[:, i], norma)

    poz = np.argmin(z)
    return poz


# def kNN(matrice, img, norma, k):
#     # Vector pentru a stoca distanțele dintre imaginea de test și fiecare imagine din matrice
#     z = np.zeros((nrPozePersTotal * nrPers))
#
#     # Calculăm norma (distanța) pentru fiecare imagine din matrice
#     for i in range(len(z)):
#         z[i] = np.linalg.norm(img - matrice[:, i], norma)
#
#     nearest_indices = np.argsort(z)[:k]
#
#     nearest_classes = [(idx // nrPozePersTotal) + 1 for idx in nearest_indices]
#
#     most_common = Counter(nearest_classes).most_common(1)[0][0]
#     return most_common


matrice = training_matrix(caleDS)

#imagine test
# cale_poza_test = f'{caleDS}/s18/8.pgm'
# test_img = cv2.imread(cale_poza_test, 0)
# test_img = np.array(test_img).reshape(-1)
# print(test_img.shape)

# Recunoaștere folosind metoda NN
# norma = 3
# predicted_index = NN(matrice, test_img, norma)

# # Reconstruim calea către imaginea prezisă
# predicted_image_path = f'{caleDS}/s{int(predicted_index / nrPozePers + 1)}/{predicted_index % nrPozePers + 1}.pgm'

# Afișăm imaginea prezisă și imaginea de test
# predicted_image = cv2.imread(predicted_image_path, 0)
# cv2.imshow('Imaginea prezisă', predicted_image)
# cv2.imshow('Imaginea de test', cv2.imread(cale_poza_test, 0))

# Recunoaștere folosind metoda kNN
k = 5  # Numărul de vecini
# predicted_class = kNN(matrice, test_img, norma, k)
# print(f'Clasa prezisă folosind kNN pentru imaginea de test este: Persoana {predicted_class}')

# Încărcăm datele și etichetele
data, etichete = load_data(caleDS)

# Setăm norma L2 (distanța Euclidiană) și numărul de vecini pentru kNN
norma = 2
k = 5  # Numărul de vecini pentru kNN

# Implementare cross-validation cu k-fold (de exemplu, k=5)
kf = KFold(n_splits=5, shuffle=True, random_state=42)


# Variabile pentru a stoca acuratețea pentru NN și kNN
accuracies_NN = []
accuracies_kNN = []
#
# Începem procesul de cross-validation
for train_index, test_index in kf.split(data):
    # Împărțim datele în set de antrenare și testare
    X_train, X_test = data[train_index], data[test_index]
    y_train, y_test = etichete[train_index], etichete[test_index]

    X_train = X_train.T
    X_test = X_test.T

    # Evaluăm metoda NN pe fiecare imagine de test
    correct_NN = 0
    correct_kNN = 0
    for i in range(X_test.shape[1]):  # Iterăm prin fiecare coloană
        test_image = X_test[:, i]
        # Recunoaștere cu NN
        predicted_index_NN = NN(X_train, test_image, norma)
        predicted_class_NN = y_train[predicted_index_NN]
        if predicted_class_NN == y_test[i]:
            correct_NN += 1

        # Recunoaștere cu kNN
        # predicted_class_kNN = kNN(X_train, test_image, y_train, norma, k)
        # if predicted_class_kNN == y_test[i]:
        #     correct_kNN += 1

    # Calculăm acuratețea pentru fiecare fold
    accuracy_NN = correct_NN / len(X_test)
    accuracy_kNN = correct_kNN / len(X_test)

    accuracies_NN.append(accuracy_NN)
    accuracies_kNN.append(accuracy_kNN)

# Afișăm rezultatele cross-validation
print(f'Acuratețea medie pentru NN: {np.mean(accuracies_NN) * 100:.4f}%')
print(f'Acuratețea medie pentru kNN (k={k}): {np.mean(accuracies_kNN) * 100:.2f}%')

# #

# Apelăm funcția cu algoritmul NN
# rata, timp_mediu = statistici(NN, matrice, test_img, etichete_test, norma=2)
# print(rata, timp_mediu)

# Sau cu kNN
# rata, timp_mediu = compute_recognition_statistics(kNN, matrice_antrenare, imagini_test, etichete_test, norma=2, k=5)



# Așteptăm ca utilizatorul să închidă ferestrele
cv2.waitKey()
cv2.destroyAllWindows()

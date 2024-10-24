import time

import cv2
import numpy as np
import random
from collections import Counter

random.seed(42)

cale_DS = r'C:\Users\Popescu Sebastian\Desktop\PatternRecogn\att_faces'

nr_pers = 40
nr_poze_pers = 10
nr_poze_test = int(input("Cum impartim cele 10 poze? Cate sa fie poze de test?"))
nr_poze_train = nr_poze_pers - nr_poze_test

rezolutie = 112 * 92


def load_data(cale_DS):
    # matrice_poze are toate pozele
    # etichete are toate nr persoanelor

    matrice_poze = np.zeros((rezolutie, nr_pers * nr_poze_pers))
    etichete = []

    for nr in range(1, nr_pers + 1):
        for im in range(1, nr_poze_pers + 1):
            img = cv2.imread(f'{cale_DS}/s{nr}/{im}.pgm', 0)
            if img is not None:
                img = np.array(img)
                img_vect = np.reshape(img, (-1))  # convertim in vector
                matrice_poze[:, (nr - 1) * nr_poze_pers + im - 1] = img_vect
                etichete.append(nr)  # fiecare poza primeste nr persoanei

    return np.array(matrice_poze), np.array(etichete)



def NN(matrice, poza_test, norma):
    z = np.zeros(matrice.shape[1])
    # print(f"poza_test shape: {poza_test.shape}")

    for i in range(matrice.shape[1]):
        # print(f"Comparing to image {i} with shape {matrice[:, i].shape}")
        if norma == 1 or norma == 2:
            z[i] = np.linalg.norm(poza_test - matrice[:, i], norma)
            # print(f"z[{i}] = {z[i]}")  # Debug print
        elif norma == 3:
            z[i] = np.linalg.norm(poza_test - matrice[:, i], np.inf)
        elif norma == 4:
            z[i] = (1 - np.dot(matrice[:, i], poza_test) // (np.linalg.norm(poza_test)))

    pozitie = np.argmin(z)
    return pozitie


def kNN(matrice, poza_test, norma, k):
    # Vector pentru a stoca diferentele dintre imaginea de test și fiecare imagine din matrice
    z = np.zeros(matrice.shape[1])
    poza_test = poza_test.flatten()

    # Calculăm norma (distanța) pentru fiecare imagine din matrice
    for i in range(matrice.shape[1]):
        if norma == 1 or norma == 2:
            z[i] = np.linalg.norm(poza_test - matrice[:, i], norma)
        elif norma == 3:
            z[i] = np.linalg.norm(poza_test - matrice[:, i], np.inf)
        elif norma == 4:
            z[i] = (1 - np.dot(matrice[:, i], poza_test) // (np.linalg.norm(poza_test)))

    indicii_apropiati = np.argsort(z)[:k]

    clasele_apropiate = [(idx // nr_poze_pers) + 1 for idx in indicii_apropiati]

    clasa_populara = Counter(clasele_apropiate).most_common(1)[0][0]
    return clasa_populara


def impartire_matrice(matrice_poze, etichete, nr_poze_test, nr_poze_train):
    matrice_train = np.zeros((rezolutie, nr_pers * nr_poze_train))
    matrice_test = np.zeros((rezolutie, nr_pers * nr_poze_test))

    etichete_train = []
    etichete_test = []

    indici_test = []

    for pers in range(nr_pers):
        # alegem 2 poze de test random
        poze_test_random = random.sample(range(nr_poze_pers), nr_poze_test)

        train_col = 0
        test_col = 0
        for im in range(nr_poze_pers):
            idx = pers * nr_poze_pers + im # indexul fiecarei imagini incepand cu 0!

            if im in poze_test_random:
                matrice_test[:, pers * nr_poze_test + test_col] = matrice_poze[:, idx] #append la matricea de test cu poza de la indexul respectiv din matricea cu toate pozele
                etichete_test.append(etichete[idx]) #la fel si cu etichetele
                test_col += 1
            else:
                matrice_train[:, pers * nr_poze_train + train_col] = matrice_poze[:, idx]
                etichete_train.append(etichete[idx])

    return matrice_train, np.array(etichete_train), matrice_test, np.array(etichete_test)


def afiseaza_poza(vector_poza, titlu="Imagine"):
    img = np.reshape(vector_poza, (112, 92))  # Reshape the vector back into 2D image
    cv2.imshow(titlu, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def statistici(alg, etichete_test, matrice_test,etichete_train, matrice_train, norma, k=None):
    nr_recunoasteri_corecte = 0
    timpi_executie = []

    for i, imagine_test in enumerate(matrice_test.T):
        start_time = time.perf_counter()

        if k is None:
            index_prezis = alg(matrice_train, imagine_test, norma) # apeleaza NN
            clasa_prezisa = etichete_train[index_prezis]
        else:
            clasa_prezisa = alg(matrice_train, imagine_test, norma, k)
            # index_prezis = random.sample(matrice_poze[:, clasa_prezisa], 1) # luam o poza random din clasa(persoana) prezisa

        end_time = time.perf_counter()
        timpi_executie.append(end_time - start_time)

        if clasa_prezisa == etichete_test[i]:
            nr_recunoasteri_corecte += 1

    rata_recunoastere = nr_recunoasteri_corecte / len(etichete_test)
    timp_mediu = np.mean(timpi_executie)

    return rata_recunoastere, timp_mediu




# Load data
matrice_poze, etichete = load_data(cale_DS)

# Split the data into train and test sets
matrice_train, etichete_train, matrice_test, etichete_test = impartire_matrice(matrice_poze, etichete, nr_poze_test, nr_poze_train)


norma = int(input("Care sa fie norma? (1/2/3/4)"))

# Run NN statistics
rata_recunoastere, timp_mediu = statistici(NN, etichete_test, matrice_test,etichete_train, matrice_train, norma)

print(f"Rata de recunoastere: {rata_recunoastere * 100:.2f}%")
print(f"Timp mediu de interogare: {timp_mediu:.6f} secunde")

# Optionally, you can test the kNN algorithm by passing k value
rata_recunoastere_knn, timp_mediu_knn = statistici(kNN, etichete_test, matrice_test,etichete_train, matrice_train, norma, k=5)
print(f"kNN Rata de recunoastere: {rata_recunoastere_knn * 100:.2f}%")
print(f"kNN Timp mediu de interogare: {timp_mediu_knn:.6f} secunde")






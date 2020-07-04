# IMPORT
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

"""
# Classe permettant transformer notre base d'images en tableaux Numpy
"""


def launchConversion(pathData, pathNumpy, resizeImg, imgSize):
    """
    # Convertit des images en tableau numpy
    :param pathData: chemin ou sont les images
    :param pathNumpy: chemin ou sera enregistré le tableau Numpy
    :param resizeImg:
    :param imgSize:
    """

    #Pour chaque classe
    for champiClasse in os.listdir(pathData):
        pathChampi = pathData + '\\' + champiClasse
        imgs = []

        #Pour chaque image d'une classe, on la charge, resize et transforme en tableau
        for imgChampi in tqdm(os.listdir(pathChampi), "Conversion de la classe : '{}'".format(champiClasse)):
            imgChampiPath = pathChampi + '\\' + imgChampi
            img = Image.open(imgChampiPath)
            img.load()
            if resizeImg == True:
                img = img.resize(size=imgSize)

            data = np.asarray(img, dtype=np.float32)
            imgs.append(data)

        #Converti les gradients de pixels (allant de 0 à 255) vers des gradients compris entre 0 et 1
        imgs = np.asarray(imgs) / 255.

        #Enregistre une classe entiere en un fichier numpy
        np.save(pathNumpy + '\\ ' + champiClasse + '.npy', imgs)


def main():
    """
    # Fonction main
    """

    pathNumpy = '.\\numpy'
    pathData = '.\\dataset'
    resizeImg = True
    imgSize = (50, 50)
    launchConversion(pathData, pathNumpy, resizeImg, imgSize)


if __name__ == '__main__':
    """
    # MAIN
    """
    main()


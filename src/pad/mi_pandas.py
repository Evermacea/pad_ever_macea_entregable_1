import pandas as pd
import numpy as np
from numpy.random import randn

class Matrices(): # Una sola dimensión
    def __init__(self):
        np.random.seed(101)

    def series(self):
        etiquetas = ['a', 'b', 'c', 'd', 'e']
        valores = [1, 3, 5, 7, 9]
        arr = np.array(valores)
        directorios = {"amiiboSeries":"Mario Sports Superstars","character":"Metal Mario","numero":123} 
        mi_serie = pd.Series(valores, index=etiquetas)
        print(etiquetas)
        print(valores)
        print(arr)
        print(directorios)
        print(mi_serie)
        
    def matrices(self, filas=0, columnas=0):
        data = pd.DataFrame(randn(filas, columnas), columns='W X Y Z'.split(" ")) # Hacer el calculo ((w*y)-z)+y
        print(data)
        data['iudigital'] = 'pad'
        data['function']= (data['W']*data['Y'] -data['Z'] )+data['Y']
        data.to_excel("datos_generados.xlsx", sheet_name="Sheet1")

matr = Matrices()
matr.series()
matr.matrices(5, 4)
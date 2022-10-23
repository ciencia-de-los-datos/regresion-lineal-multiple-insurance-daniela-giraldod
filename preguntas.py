"""
Regresión Lineal Multiple
-----------------------------------------------------------------------------------------

En este laboratorio se entrenara un modelo de regresión lineal multiple que incluye la 
selección de las n variables más relevantes usando una prueba f.

"""
# pylint: disable=invalid-name
# pylint: disable=unsubscriptable-object

import pandas as pd

def read_data(input):
    datos = pd.read_csv(
        input,
    )
    return datos

input="insurance.csv"


df = read_data(input)

def pregunta_01():
    """
    Carga de datos.
    -------------------------------------------------------------------------------------
    """
    # Lea el archivo `insurance.csv` y asignelo al DataFrame `df`
    df = read_data(input)

    # Asigne la columna `charges` a la variable `y`.
    y = df["charges"]

    # Asigne una copia del dataframe `df` a la variable `X`.
    X = df.copy(deep=True)

    # Remueva la columna `charges` del DataFrame `X`.
    X.drop(labels=['charges'],axis=1,inplace=True)

    # Retorne `X` y `y`
    return X, y



def pregunta_02():
    """
    Preparación de los conjuntos de datos.
    -------------------------------------------------------------------------------------
    """

    # Importe train_test_split
    from sklearn.model_selection import train_test_split

    # Cargue los datos y asigne los resultados a `X` y `y`.
    X, y = pregunta_01()

    # Divida los datos de entrenamiento y prueba. La semilla del generador de números
    # aleatorios es 12345. Use 300 patrones para la muestra de prueba.
    tamaño_prueba=(300/len(X))
    (X_train, X_test, y_train, y_test,) = train_test_split(
        X,
        y,
        test_size=tamaño_prueba,
        random_state=12345,
    )
    # Retorne `X_train`, `X_test`, `y_train` y `y_test`
    return X_train, X_test, y_train, y_test

def pregunta_03():
    """
    Especificación del pipeline y entrenamiento
    -------------------------------------------------------------------------------------
    """

    # Importe make_column_selector
    # Importe make_column_transformer
    # Importe SelectKBest
    # Importe f_regression
    # Importe LinearRegression
    # Importe GridSearchCV
    # Importe Pipeline
    # Importe OneHotEncoder
    from sklearn.compose import make_column_selector


    from sklearn.compose import make_column_selector
    from sklearn.compose import make_column_transformer
    from sklearn.feature_selection import SelectKBest,f_regression
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    import numpy as np

#Un ColumnTransformer toma una lista, que contiene tuplas de las transformaciones que deseamos realizar en las diferentes columnas. 
#Cada tupla espera 3 valores separados por comas: primero, el nombre del transformador, que puede ser prácticamente cualquier cosa(pasado como una cadena),
#segundo es el objeto estimador y el último son las columnas sobre las que deseamos realizar esa operación .

     

def pregunta_04():
  

"""
    Evaluación del modelo
    -------------------------------------------------------------------------------------
    """

    # Importe mean_squared_error
    from ____ import ____

    # Obtenga el pipeline optimo de la pregunta 3.
    gridSearchCV = pregunta_03()

    # Cargue las variables.
    X_train, X_test, y_train, y_test = pregunta_02()

    # Evalúe el modelo con los conjuntos de entrenamiento y prueba.
    y_train_pred = ____.____(____)
    y_test_pred = ____.____(____)

    # Compute el error cuadratico medio de entrenamiento y prueba. Redondee los
    # valores a dos decimales.

    mse_train = ____(
        _____,
        _____,
    ).round(2)

    mse_test = ____(
        _____,
        _____,
    ).round(2)

    # Retorne el error cuadrático medio para entrenamiento y prueba
    return mse_train, mse_test

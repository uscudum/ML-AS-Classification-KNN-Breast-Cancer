# Clasificación de Tumores utilizando k-NN

Este proyecto tiene como objetivo clasificar tumores como benignos o malignos utilizando el algoritmo k-Nearest Neighbors (k-NN). El código ha sido implementado en Google Colab para facilitar su ejecución y modificación en la nube.

## Descripción del Dataset

El dataset utilizado en este proyecto contiene características de imágenes de tumores, que incluyen medidas como el radio medio, la textura media, el perímetro medio, y el área media de los núcleos de las células tumorales. Cada entrada en el dataset está etiquetada como "Benigno" (B) o "Maligno" (M).

## Estructura del Código

### 1. Carga del Dataset
```python
df = pd.read_csv('data.csv')
df.describe()
```

Se carga el dataset data.csv en un DataFrame de pandas y se muestran las estadísticas descriptivas de las características.

### 2. Selección de Características
```python
X_train = df[['radius_mean', 'texture_mean', 'perimeter_mean','area_mean']]
y = df['diagnosis'].replace({'B': 0, 'M': 1})
```

Se seleccionan las columnas relevantes para el entrenamiento del modelo, y se transforma la columna de diagnóstico (diagnosis) en valores numéricos: 0 para benigno y 1 para maligno.

### 3. Normalización de Características
```python
scaler = StandardScaler()
X = scaler.fit_transform(X_train)
```

Se normalizan las características seleccionadas utilizando StandardScaler para que tengan media 0 y desviación estándar 1.

### 4. Creación y Entrenamiento del Modelo k-NN
```python
Kn = KNeighborsClassifier(n_neighbors=3)
Kn.fit(X, y)
```

Se crea un modelo k-NN con 3 vecinos y se entrena con las características normalizadas.

### 5. Clasificación Manual
```python
mean_radius = float(input("Ingrese mean radius: "))
mean_texture = float(input("Ingrese mean texture: "))
mean_perimeter = float(input("Ingrese mean perimeter: "))
mean_area = float(input("Ingrese mean area: "))

input_origin = [[mean_radius, mean_texture, mean_perimeter, mean_area]]
input_std = scaler.transform(input_origin)
prediccion = Kn.predict(input_std)
```

El usuario puede ingresar manualmente las características del tumor que desea clasificar. Estas características se normalizan y se predice si el tumor es benigno o maligno.

### 6. Interpretación del Resultado
```python
if prediccion[0] == 0:
    print("El tumor es benigno.")
else:
    print("El tumor es maligno.")
```

Finalmente, se interpreta la predicción del modelo, mostrando si el tumor es benigno o maligno.

## Requisitos
* Google Colab
* Pandas
* Scikit-learn

## Ejecución en Google Colab

Para ejecutar este proyecto, sigue los siguientes pasos:

1. Sube el archivo data.csv a tu entorno de Google Colab.
2. Copia y pega el código proporcionado en una celda de código en Google Colab.
3. Ejecuta las celdas para cargar el dataset, entrenar el modelo y realizar la clasificación manual.

## Notas
* El dataset debe estar disponible como data.csv en el mismo directorio donde se ejecuta el código.
* Este proyecto es ideal para entender el proceso de clasificación utilizando k-NN, así como para aprender sobre la normalización de datos y la importancia de seleccionar características relevantes.


# # # # # # # # # # # # #
# Reconocedor de letras #
# # # # # # # # # # # # #

# CONFIGURAR EL ENTORNO VIRTUAL
# python3.12 -m venv venv
# source venv/bin/activate
# python -m pip install --upgrade pip
# python -m pip install opencv-python
# python -m pip install tensorflow numpy
# python -m pip install tensorflow-datasets==4.9.9
# python -m pip install importlib-resources

# VOLVER A CORRER
# cd ~/Desktop
# source venv/bin/activate
# python reconocedor_letras.py

# IMPORTAR LIBRERIAS
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import cv2


def reconocer_letra(modelo, imagen_28):
    prediccion = modelo.predict(
        np.expand_dims(imagen_28, axis=0),
        verbose=0
    )
    indice = np.argmax(prediccion[0])
    letra_reconocida = chr(ord("A") + indice)
    confianza = prediccion[0][indice]
    return letra_reconocida, float(confianza)

# CARGA DE EMNIST
# Cargamos las imágenes de entrenamiento y prueba de EMNIST Letters.
(ds_train, ds_test), ds_info = tfds.load(
    "emnist/letters",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

def preparar_imagen(imagen, etiqueta):
    imagen = tf.cast(imagen, tf.float32)
    imagen = imagen / 255.0
    etiqueta = etiqueta - 1
    return imagen, etiqueta

# Aplicamos la función a todas las imágenes de entrenamiento.
ds_train = ds_train.map(preparar_imagen)
# Aplicamos la función a todas las imágenes de prueba.
ds_test = ds_test.map(preparar_imagen)

# ENTRENAMIENTO
# Agrupamos las imágenes de entrenamiento en grupos de 128.
ds_train = ds_train.batch(128)
# Agrupamos las imágenes de prueba en grupos de 128.
ds_test = ds_test.batch(128)

# RED NEURONAL

modelo = tf.keras.Sequential([
    # Convertimos cada imagen 28x28x1 en un vector.
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    # Primera capa neuronal.
    tf.keras.layers.Dense(128, activation="relu"),
    # Segunda capa neuronal.
    tf.keras.layers.Dense(64, activation="relu"),
    # Capa de salida.
    # Hay 26 neuronas porque tenemos 26 letras.
    tf.keras.layers.Dense(26, activation="softmax")
])

# ENTRENAMIENTO

# Configuramos cómo aprenderá la red neuronal.
modelo.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
print("Comenzando entrenamiento...")
# Entrenamos la red durante 5 épocas.
modelo.fit(
    ds_train,
    epochs=5
)

# EVALUACIO DEL MODELO
print("Evaluando el modelo...")
# Evaluamos el modelo utilizando las imágenes de prueba.
resultado = modelo.evaluate(ds_test)
# Obtenemos la precisión.
precision = resultado[1]
# Mostramos la precisión.
print("Precisión del modelo:", precision)

# USO CON CAMARA DE PC
camara = cv2.VideoCapture(0)
if not camara.isOpened():
    print("ERROR: No se pudo abrir la cámara.")
    exit()
print()

print("======================================")
print("   RECONOCEDOR DE LETRAS")
print("======================================")
print("Escribe una letra dentro del cuadro.")
print("Presiona ESPACIO para reconocerla.")
print("Presiona ESC para salir.")
print()

while True:
    exito, imagen = camara.read()
    if not exito:
        print("No se pudo obtener imagen de la cámara.")
        break

    # Obtenemos el tamaño de la imagen.
    alto, ancho, _ = imagen.shape
    # Definimos el tamaño del cuadro.
    tamano = 300
    # Calculamos la posición horizontal del cuadro.
    x1 = (ancho - tamano) // 2
    # Calculamos la posición vertical del cuadro.
    y1 = (alto - tamano) // 2
    # Calculamos el segundo punto horizontal.
    x2 = x1 + tamano
    # Calculamos el segundo punto vertical.
    y2 = y1 + tamano
    # Dibujamos el cuadro en la imagen.
    cv2.rectangle(
        imagen,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Escribimos instrucciones en la ventana.
    cv2.putText(
        imagen,
        "Escribe una letra dentro del cuadro",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # Mostramos otra instrucción.
    cv2.putText(
        imagen,
        "ESPACIO = reconocer | ESC = salir",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    # Mostramos la imagen en una ventana.
    cv2.imshow(
        "Reconocedor de letras",
        imagen
    )

    # Esperamos 1 milisegundo para detectar una tecla.
    tecla = cv2.waitKey(1) & 0xFF

    # Si se presiona ESC...
    if tecla == 27:

        # Salimos del ciclo.
        break

        # Si se presiona la barra espaciadora...
    if tecla == 32:

        # Recortamos únicamente el cuadro central.
        recorte = imagen[y1:y2, x1:x2]

        # Convertimos la imagen a escala de grises.
        gris = cv2.cvtColor(
            recorte,
            cv2.COLOR_BGR2GRAY
        )

        # Aplicamos un umbral.
        # Los píxeles oscuros se vuelven negros
        # y los claros se vuelven blancos.
        _, binaria = cv2.threshold(
            gris,
            120,
            255,
            cv2.THRESH_BINARY_INV
        )

        # Buscamos los contornos de la letra.
        contornos, _ = cv2.findContours(
            binaria,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Comprobamos que encontramos algo.
        if len(contornos) > 0:

            # Seleccionamos el contorno más grande.
            contorno = max(
                contornos,
                key=cv2.contourArea
            )

            # Obtenemos el rectángulo de la letra.
            x, y, w, h = cv2.boundingRect(contorno)

            # Recortamos únicamente la letra.
            letra_recorte = binaria[
                y:y+h,
                x:x+w
            ]

            # Calculamos el tamaño máximo.
            lado = max(w, h)

            # Creamos una imagen negra cuadrada.
            cuadrado = np.zeros(
                (lado, lado),
                dtype=np.uint8
            )

            # Calculamos la posición para centrar
            # la letra horizontalmente.
            inicio_x = (lado - w) // 2

            # Calculamos la posición para centrar
            # la letra verticalmente.
            inicio_y = (lado - h) // 2

            # Colocamos la letra en el centro.
            cuadrado[
                inicio_y:inicio_y+h,
                inicio_x:inicio_x+w
            ] = letra_recorte

            # Redimensionamos a 20x20.
            letra_20 = cv2.resize(
                cuadrado,
                (20, 20)
            )

            # Creamos una imagen de 28x28.
            imagen_28 = np.zeros(
                (28, 28),
                dtype=np.uint8
            )

            # Colocamos la letra de 20x20
            # en el centro de la imagen.
            imagen_28[
                4:24,
                4:24
            ] = letra_20

            # EMNIST tiene una orientación diferente.
            imagen_28 = np.transpose(
                imagen_28
            )

            # Invertimos verticalmente.
            imagen_28 = np.flipud(
                imagen_28
            )

            # Convertimos a float32.
            imagen_28 = imagen_28.astype(
                "float32"
            )

            # Normalizamos entre 0 y 1.
            imagen_28 = imagen_28 / 255.0

            # Guardamos la imagen que verá la red neuronal.
            cv2.imwrite(
                "imagen_que_ve_la_red.png",
                (imagen_28 * 255).astype(np.uint8)
            )

            print("Imagen guardada como imagen_que_ve_la_red.png")

            # Agregamos la dimensión del canal.
            imagen_28 = np.expand_dims(
                imagen_28,
                axis=-1
            )

            # Agregamos la dimensión del lote.
            imagen_28 = np.expand_dims(
                imagen_28,
                axis=0
            )

            print("\nImagen 28x28 que recibe la red:")

            for fila in imagen_28[0, :, :, 0]:
                print(
                    "".join(
                        "#" if pixel > 0.2 else " "
                        for pixel in fila
                    )
                )

            letra, confianza = reconocer_letra(modelo, imagen_28[0])

            # Mostramos el resultado.
            print()
            print("------------------------------")
            print(
                "Letra reconocida:",
                letra
            )
            print(
                "Confianza:",
                round(
                    float(confianza) * 100,
                    2
                ),
                "%"
            )
            print("------------------------------")

            # Mostramos la letra reconocida.
            cv2.putText(
                imagen,
                "Letra: " + letra,
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

            # Mostramos nuevamente la cámara.
            cv2.imshow(
                "Reconocedor de letras",
                imagen
            )

        else:

            print("No se detecto ninguna letra.")

# Liberamos la cámara.
camara.release()
cv2.destroyAllWindows()
print("Programa terminado.")
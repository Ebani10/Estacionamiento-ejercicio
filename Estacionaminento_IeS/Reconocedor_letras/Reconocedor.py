import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def preparar_imagen(imagen, etiqueta):
    imagen = tf.cast(imagen, tf.float32) / 255.0
    return imagen, etiqueta - 1


def crear_modelo_entrenado():
    (ds_train, ds_test), _ = tfds.load(
        "emnist/letters",
        split=["train", "test"],
        as_supervised=True,
        with_info=True
    )
    ds_train = ds_train.map(preparar_imagen).batch(128)
    ds_test = ds_test.map(preparar_imagen).batch(128)

    modelo = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(26, activation="softmax")
    ])
    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    modelo.fit(ds_train, epochs=5, verbose=1)
    modelo.evaluate(ds_test, verbose=1)
    return modelo


def reconocer_letra(modelo, imagen_28):
    prediccion = modelo.predict(
        np.expand_dims(imagen_28, axis=0),
        verbose=0
    )
    indice = np.argmax(prediccion[0])
    return chr(ord("A") + indice), float(prediccion[0][indice])


def preparar_captura(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, 120, 255, cv2.THRESH_BINARY_INV)
    contornos, _ = cv2.findContours(
        binaria,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    if not contornos:
        return None

    contorno = max(contornos, key=cv2.contourArea)
    x, y, ancho, alto = cv2.boundingRect(contorno)
    letra_recorte = binaria[y:y + alto, x:x + ancho]
    lado = max(ancho, alto)
    cuadrado = np.zeros((lado, lado), dtype=np.uint8)
    inicio_x = (lado - ancho) // 2
    inicio_y = (lado - alto) // 2
    cuadrado[inicio_y:inicio_y + alto, inicio_x:inicio_x + ancho] = letra_recorte
    letra_20 = cv2.resize(cuadrado, (20, 20))

    imagen_28 = np.zeros((28, 28), dtype=np.float32)
    imagen_28[4:24, 4:24] = letra_20
    imagen_28 = np.flipud(np.transpose(imagen_28)) / 255.0
    return np.expand_dims(imagen_28, axis=-1)


def reconocer_placa():
    modelo = crear_modelo_entrenado()
    camara = cv2.VideoCapture(0)
    if not camara.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    try:
        while True:
            exito, imagen = camara.read()
            if not exito:
                raise RuntimeError("No se pudo obtener imagen de la cámara.")

            alto, ancho, _ = imagen.shape
            tamano = 300
            x1 = (ancho - tamano) // 2
            y1 = (alto - tamano) // 2
            x2, y2 = x1 + tamano, y1 + tamano
            cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                imagen,
                "Escribe una letra | ESPACIO = reconocer | ESC = salir",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            cv2.imshow("Reconocedor de letras", imagen)
            tecla = cv2.waitKey(1) & 0xFF

            if tecla == 27:
                return None
            if tecla != 32:
                continue

            imagen_28 = preparar_captura(imagen[y1:y2, x1:x2])
            if imagen_28 is None:
                print("No se detecto ninguna letra.")
                continue

            letra, confianza = reconocer_letra(modelo, imagen_28)
            print(f"Letra reconocida: {letra}")
            print(f"Confianza: {confianza * 100:.2f}%")
            return letra
    finally:
        camara.release()
        cv2.destroyAllWindows()

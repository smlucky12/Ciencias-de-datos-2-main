import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import tempfile
from ultralytics import YOLO
from insightface.app import FaceAnalysis

import emociones
from identidad import get_identity_predictor


st.set_page_config(layout="wide", page_title="Detector de Rostros, Emociones e Identidad")

st.title("👤 Detector de Rostros, Emociones e Identidad")
st.markdown("### Proyecto Ciencia de Datos 2 - UNALM")


# Sidebar
modo = st.sidebar.radio("Selecciona Modo", ["Imagen", "Webcam"])
modelo_opcion = st.sidebar.selectbox(
    "Modelo de detección",
    ["Haar Cascade", "YOLO"]
)


# -----------------------------
# CARGA DE MODELOS
# -----------------------------

@st.cache_resource
def cargar_haar():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


@st.cache_resource
def cargar_yolo():
    weights = "yolo11n-face.pt"

    if not os.path.exists(weights):
        import urllib.request
        url = "https://huggingface.co/deepghs/yolo-face/resolve/main/yolov11n-face/model.pt?download=true"
        urllib.request.urlretrieve(url, weights)

    return YOLO(weights)


@st.cache_resource
def cargar_face_app():
    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU
    return face_app


@st.cache_resource
def cargar_predictor_identidad():
    face_app = cargar_face_app()
    return get_identity_predictor(
        modelos_dir="modelos_identidad",
        face_app=face_app
    )


# -----------------------------
# FUNCIONES AUXILIARES
# -----------------------------

def predecir_identidad_desde_recorte(rostro_recortado):
    predictor = cargar_predictor_identidad()

    if rostro_recortado is None or rostro_recortado.size == 0:
        return "Sin rostro", 0.0

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        ruta_temp = tmp.name
        cv2.imwrite(ruta_temp, rostro_recortado)

    nombre, conf_id, bbox = predictor(ruta_temp)

    try:
        os.remove(ruta_temp)
    except:
        pass

    if nombre is None:
        nombre = "Sin rostro"

    return nombre, conf_id


def dibujar_label(frame, texto, x, y, color):
    y_text = max(y - 10, 25)
    cv2.putText(
        frame,
        texto,
        (x, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )


# -----------------------------
# DETECCIÓN + EMOCIÓN + IDENTIDAD
# -----------------------------

def detectar_y_clasificar(frame):
    count = 0

    # 🔥 PREDICCIÓN DE IDENTIDAD UNA SOLA VEZ (imagen completa)
    predictor = cargar_predictor_identidad()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        ruta_temp = tmp.name
        cv2.imwrite(ruta_temp, frame)

    nombre_global, conf_id, bbox_id = predictor(ruta_temp)

    try:
        os.remove(ruta_temp)
    except:
        pass

    if nombre_global is None:
        nombre_global = "Desconocido"

    # -------------------------
    # DETECCIÓN NORMAL
    # -------------------------

    if modelo_opcion == "Haar Cascade":
        detector = cargar_haar()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            rostro_recortado = frame[y:y+h, x:x+w]

            emo, conf_emo = emociones.predecir_emocion(rostro_recortado)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            label = f"{nombre_global} | {emo} ({conf_emo:.1f}%)"
            dibujar_label(frame, label, x, y, (255, 255, 255))

            count += 1

    else:
        model = cargar_yolo()
        res = model(frame, verbose=False)[0]

        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            conf_det = float(b.conf[0])

            if conf_det > 0.5:
                rostro_recortado = frame[y1:y2, x1:x2]

                emo, conf_emo = emociones.predecir_emocion(rostro_recortado)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                label = f"{nombre_global} | {emo} ({conf_emo:.1f}%)"
                dibujar_label(frame, label, x1, y1, (0, 255, 255))

                count += 1

    return frame, count
# -----------------------------
# INTERFAZ
# -----------------------------

if modo == "Imagen":
    uploaded_file = st.file_uploader(
        "Sube una imagen de un cantante",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        st.image(image, caption="Imagen original", use_column_width=True)

        if st.button("Analizar Imagen"):
            with st.spinner("Detectando rostros, emociones e identidad..."):
                res_frame, total = detectar_y_clasificar(frame)

            st.success(f"Se detectaron {total} rostros.")
            st.image(
                cv2.cvtColor(res_frame, cv2.COLOR_BGR2RGB),
                caption="Resultado",
                use_column_width=True
            )


else:
    run = st.checkbox("Iniciar cámara en vivo")
    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()

            if not ret:
                st.error("No se pudo acceder a la cámara")
                break

            frame, total = detectar_y_clasificar(frame)

            cv2.putText(
                frame,
                f"Rostros: {total}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
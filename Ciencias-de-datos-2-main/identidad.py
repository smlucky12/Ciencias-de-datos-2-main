import os
import cv2
import numpy as np
import pickle
import warnings
from PIL import Image

warnings.filterwarnings("ignore")


def predecir_identidad(ruta_imagen, face_app, clf, scaler, label_encoder, umbral=0.40):
    img = cv2.imread(ruta_imagen)
    if img is None:
        img = cv2.cvtColor(np.array(Image.open(ruta_imagen).convert("RGB")), cv2.COLOR_RGB2BGR)
    faces = face_app.get(img)
    if not faces: return None, 0.0, None
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    emb  = scaler.transform(face.embedding.reshape(1,-1))
    if hasattr(clf,"predict_proba"):
        proba = clf.predict_proba(emb)[0]; idx = np.argmax(proba); conf = proba[idx]
    else:
        idx = clf.predict(emb)[0]; conf = 1.0
    nombre = label_encoder.inverse_transform([idx])[0] if conf >= umbral else "Desconocido"
    return nombre, float(conf), face.bbox.astype(int)

def visualizar_prediccion(ruta, nombre, confianza, bbox):
    img = cv2.imread(ruta)
    if img is None:
        img = cv2.cvtColor(np.array(Image.open(ruta).convert("RGB")), cv2.COLOR_RGB2BGR)
    if bbox is not None:
        x1,y1,x2,y2 = bbox
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,200,50),2)
        cv2.putText(img,f"{nombre} ({confianza:.0%})",(x1,y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,200,50),2)


def get_identity_predictor(modelos_dir="modelos_identidad", face_app=None):
    with open(f"{modelos_dir}/scaler.pkl","rb") as f: sc = pickle.load(f)
    with open(f"{modelos_dir}/label_encoder.pkl","rb") as f: encoder = pickle.load(f)
    pkls = [f for f in os.listdir(modelos_dir)
            if f.endswith(".pkl") and f not in ("scaler.pkl","label_encoder.pkl")]
    with open(f"{modelos_dir}/{pkls[0]}","rb") as f: clf_c = pickle.load(f)
    def predictor(ruta_img):
        return predecir_identidad(ruta_img, face_app, clf_c, sc, encoder)
    print(f"✅ Predictor listo — modelo: {pkls[0]}")
    return predictor

print("Función get_identity_predictor() disponible para siguiente paso ✅")
import cv2
from deepface import DeepFace
import numpy as np

def predecir_emocion(imagen_cargada):
    """
    Esta función recibe la imagen que el usuario subió a la web.
    """
    try:
        # Imagen cargada
        analisis = DeepFace.analyze(
            img_path = imagen_cargada, 
            actions = ['emotion'],
            enforce_detection = False,
            silent = True
        )
        
        # Emoción dominante
        emocion = analisis[0]['dominant_emotion']
        confianza = analisis[0]['emotion'][emocion]
        
        return emocion, confianza
    except Exception as e:
        return "Error al procesar", 0.0
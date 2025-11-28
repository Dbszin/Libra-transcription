import cv2
import numpy as np
import streamlit as st
import os
from PIL import Image
import warnings

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

mp_hands = mp.solutions.hands

RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
}

script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, "..", "modelo_treinado.h5")
path = os.path.normpath(path)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = tf.keras.models.load_model(path, compile=False)



CONFIDENCE_THRESHOLD = 0.25
DEBUG_MODE = False


def predict_object(hand_roi, show_top3=False):
    if hand_roi.size == 0 or hand_roi.shape[0] < 10 or hand_roi.shape[1] < 10:
        return "Mão não detectada"
    
    resized_hand = cv2.resize(hand_roi, (224, 224), interpolation=cv2.INTER_AREA)
    normalized_hand = resized_hand.astype('float32') / 255.0
    
    predictions = model.predict(np.expand_dims(normalized_hand, axis=0), verbose=0)
    predicted_class = predictions.argmax()
    confidence = predictions[0][predicted_class]
    
    if show_top3:
        top3_indices = np.argsort(predictions[0])[-3:][::-1]
        top3_predictions = [(label_to_text[i], f"{predictions[0][i]:.3f}") for i in top3_indices]
        print(f"Top 3 predições: {top3_predictions}")
    
    if confidence < CONFIDENCE_THRESHOLD:
        return "Confiança baixa"
    
    return label_to_text[predicted_class]


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def coordinates(self, hand_landmarks, img):
        offset = 40
        
        x_min = max(0, int(min([landmark.x for landmark in hand_landmarks.landmark]) * img.shape[1] - offset))
        x_max = min(img.shape[1], int(max([landmark.x for landmark in hand_landmarks.landmark]) * img.shape[1] + offset))
        y_min = max(0, int(min([landmark.y for landmark in hand_landmarks.landmark]) * img.shape[0] - offset))
        y_max = min(img.shape[0], int(max([landmark.y for landmark in hand_landmarks.landmark]) * img.shape[0] + offset))
        
        if x_max <= x_min or y_max <= y_min:
            return
        
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        hand_roi = img[y_min:y_max, x_min:x_max]
        
        if hand_roi.size > 0 and hand_roi.shape[0] > 10 and hand_roi.shape[1] > 10:
            global DEBUG_MODE
            predicted_object = predict_object(hand_roi, show_top3=DEBUG_MODE)
            
            text_x = x_min
            text_y = max(0, y_min - 10)
            
            (text_width, text_height), baseline = cv2.getTextSize(predicted_object, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(img, (text_x, text_y - text_height - baseline), 
                         (text_x + text_width, text_y + baseline), (0, 0, 0), -1)
            cv2.putText(img, predicted_object, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.coordinates(hand_landmarks, img)
        
        return img


def show_sign_examples():
    st.subheader("📚 Exemplos de Sinais")
    
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "dataset")
    
    if not os.path.exists(dataset_path):
        st.warning("Pasta de dataset não encontrada.")
        return
    
    classes = [d for d in os.listdir(dataset_path) 
               if os.path.isdir(os.path.join(dataset_path, d))]
    
    if not classes:
        st.warning("Nenhuma classe encontrada no dataset.")
        return
    
    tabs = st.tabs(classes)
    
    for tab, class_name in zip(tabs, classes):
        with tab:
            class_path = os.path.join(dataset_path, class_name)
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if images:
                cols = st.columns(3)
                for idx, img_name in enumerate(images[:6]):
                    with cols[idx % 3]:
                        img_path = os.path.join(class_path, img_name)
                        try:
                            img = Image.open(img_path)
                            st.image(img, caption=f"{class_name}", use_container_width=True)
                        except Exception:
                            st.error(f"Erro ao carregar: {img_name}")
            else:
                st.info(f"Nenhuma imagem encontrada para '{class_name}'")




st.sidebar.title('Reconhecimento de :red[Sinais] :wave:')

st.sidebar.info("""
## Reconhecimento de Mãos - Projeto

Este projeto visa desenvolver um programa capaz de utilizar uma rede neural treinada para detectar mãos em tempo real por meio da câmera do usuário. Usando um modelo de rede neural CNN. 
""")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Configurações")

confidence_threshold = st.sidebar.slider(
    "Threshold de Confiança",
    min_value=0.0,
    max_value=1.0,
    value=0.15,
    step=0.05,
    help="Ajuste o nível mínimo de confiança para exibir predições"
)

CONFIDENCE_THRESHOLD = confidence_threshold
DEBUG_MODE = st.sidebar.checkbox("Modo Debug", help="Mostra as top 3 predições no console")

webrtc_streamer(
    key="hand-recognition-1", 
    video_processor_factory=VideoTransformer,
    rtc_configuration=RTC_CONFIGURATION
)

if st.button("Clique aqui para exibir as imagens"):
    show_sign_examples()




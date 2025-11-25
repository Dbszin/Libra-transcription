import cv2
import numpy as np
import streamlit as st
import os

os.environ["KERAS_BACKEND"] = "tensorflow"
# Configura variáveis de ambiente do TensorFlow antes de importar
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduz logs do TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Desabilita oneDNN para evitar warnings

import tensorflow as tf
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import h5py

mp_hands = mp.solutions.hands

RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
}


# Obtém o diretório do script atual
script_dir = os.path.dirname(os.path.abspath(__file__))
# Constrói o caminho para o modelo (está na raiz do projeto)
path = os.path.join(script_dir, "..", "modelo_treinado.h5")
path = os.path.normpath(path)  # Normaliza o caminho para o sistema operacional

# Verifica se o arquivo existe, caso contrário tenta o modelo original
if not os.path.exists(path):
    print(f"⚠️  Modelo treinado não encontrado em: {path}")
    print("   Tentando carregar modelo original...")
    path = os.path.join(script_dir, "..", "reconhecimento_libras", "modelo", "NewModel.h5")
    path = os.path.normpath(path)

# Carrega o modelo usando tf.keras diretamente
# compile=False evita problemas com otimizador e é suficiente para inferência
# Usa suppress_warnings para evitar problemas com name scopes
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        model = tf.keras.models.load_model(path, compile=False)
    except (IndexError, AttributeError) as e:
        # Se houver erro de name scope, tenta com uma abordagem diferente
        print(f"Erro ao carregar modelo (tentando método alternativo): {e}")
        # Reseta o grafo do TensorFlow e tenta novamente
     #   tf.keras.backend.clear_session()
        model = tf.keras.models.load_model(path, compile=False)


# Dicionário de labels - atualizado automaticamente se modelo_treinado_labels.py existir
try:
    # Tenta importar o dicionário do modelo treinado
    import sys
    sys.path.append(os.path.join(script_dir, ".."))
    from modelo_treinado_labels import label_to_text
    print(f"✅ Labels carregados do modelo treinado: {label_to_text}")
except ImportError:
    # Fallback para o modelo original se não encontrar
    label_to_text = {0: 'bus', 1: 'bank', 2: 'car', 3: 'formation', 4: 'hospital', 5: 'I', 6: 'man', 7: 'motorcycle', 8: 'my', 9: 'supermarket', 10: 'we', 11: 'woman', 12: 'you', 13: 'you (plural)', 14: 'your'}
    print("⚠️  Usando labels do modelo original")

# Threshold de confiança (pode ser ajustado)
CONFIDENCE_THRESHOLD = 0.25  # 

# Variável global para modo debug
DEBUG_MODE = False


def predict_object(hand_roi, show_top3=False):
    # Verifica se a ROI não está vazia
    if hand_roi.size == 0 or hand_roi.shape[0] < 10 or hand_roi.shape[1] < 10:
        return "Mão não detectada"
    
    # Redimensiona para o tamanho esperado pelo modelo (120x213)
    # Usa INTER_AREA para melhor qualidade ao redimensionar
    resized_hand = cv2.resize(hand_roi, (224, 224), interpolation=cv2.INTER_AREA)
    
    # Normaliza a imagem (0-255 -> 0-1)
    normalized_hand = resized_hand.astype('float32') / 255.0
    
    # Faz a predição
    predictions = model.predict(np.expand_dims(normalized_hand, axis=0), verbose=0)
    predicted_class = predictions.argmax()
    confidence = predictions[0][predicted_class]
    
    # Se show_top3 for True, mostra as top 3 predições (útil para debug)
    if show_top3:
        top3_indices = np.argsort(predictions[0])[-3:][::-1]
        top3_predictions = [(label_to_text[i], f"{predictions[0][i]:.3f}") for i in top3_indices]
        print(f"Top 3 predições: {top3_predictions}")
    
    # Só retorna a predição se a confiança for maior que o threshold
    if confidence < CONFIDENCE_THRESHOLD:
        return "Confiança baixa"
    
    predicted_object = label_to_text[predicted_class]
    
    # Retorna apenas o nome do objeto
    return predicted_object


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        # 🚨 CORREÇÃO: Inicializa o MediaPipe para suportar 2 mãos
        self.hands = mp_hands.Hands(
            max_num_hands=2, # Suporta 2 mãos, como na coleta
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    
    def coordinates(self, hand_landmarks, img):
        # Aumenta o offset para capturar mais área ao redor da mão
        offset = 40
        
        x_min = max(0, int(min([landmark.x for landmark in hand_landmarks.landmark]) * img.shape[1] - offset))
        x_max = min(img.shape[1], int(max([landmark.x for landmark in hand_landmarks.landmark]) * img.shape[1] + offset))
        y_min = max(0, int(min([landmark.y for landmark in hand_landmarks.landmark]) * img.shape[0] - offset))
        y_max = min(img.shape[0], int(max([landmark.y for landmark in hand_landmarks.landmark]) * img.shape[0] + offset))
        
        # Verifica se a região é válida
        if x_max <= x_min or y_max <= y_min:
            return
        
        # Desenha o retângulo ao redor da mão
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Extrai a região de interesse (ROI) da mão
        hand_roi = img[y_min:y_max, x_min:x_max]
        
        # Verifica se a ROI é válida antes de fazer a predição
        if hand_roi.size > 0 and hand_roi.shape[0] > 10 and hand_roi.shape[1] > 10:
            # Usa o modo debug se estiver habilitado
            global DEBUG_MODE
            predicted_object = predict_object(hand_roi, show_top3=DEBUG_MODE)
            
            text_x = x_min
            text_y = max(0, y_min - 10)
            
            # Desenha o texto com fundo preto para melhor legibilidade
            (text_width, text_height), baseline = cv2.getTextSize(predicted_object, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(img, (text_x, text_y - text_height - baseline), 
                         (text_x + text_width, text_y + baseline), (0, 0, 0), -1)
            img = cv2.putText(img, predicted_object, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        

    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")



        img = cv2.flip(img, 1)

       

        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb_frame)

       

        if results.multi_hand_landmarks:

            fist_hand = results.multi_hand_landmarks[0]

            self.coordinates(fist_hand, img)

           

            #Check if there is a second hand

            if len(results.multi_hand_landmarks) > 1:

                second_hand = results.multi_hand_landmarks[1]

                self.coordinates(second_hand, img)

           

        return img

st.sidebar.image("https://www.mjvinnovation.com/wp-content/uploads/2021/07/mjv_blogpost_redes_neurais_ilustracao_cerebro-01-1024x1020.png")

st.sidebar.title('Reconhecimento de :red[Sinais] :wave:')


st.sidebar.info("""
## Reconhecimento de Mãos - Projeto

Este projeto visa desenvolver um programa capaz de utilizar uma rede neural treinada para detectar mãos em tempo real por meio da câmera do usuário. Usando um modelo de rede neural CNN. 
""")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Configurações")

# Controle de threshold de confiança
confidence_threshold = st.sidebar.slider(
    "Threshold de Confiança",
    min_value=0.0,
    max_value=1.0,
    value=0.15,
    step=0.05,
    help="Ajuste o nível mínimo de confiança para exibir predições"
)

# Atualiza o threshold global e modo debug
CONFIDENCE_THRESHOLD = confidence_threshold
DEBUG_MODE = st.sidebar.checkbox("Modo Debug", help="Mostra as top 3 predições no console")

def exibir_imagem():
    st.subheader("Imagem dos sinais")
    num_colunas = 5
    imagens = [
        'assets/bank_1605967468_148.jpeg',
        'assets/bus_1605967420_87.jpeg',
        'assets/car_1605967469_166.jpeg',
        'assets/formation_1605967420_969.jpeg',
        'assets/hospital_1605967420_62.jpeg',
        'assets/I_1605967469_110.jpeg',
        'assets/man_1605967420_82.jpeg',
        'assets/motorcycle_1605967415_6.jpeg',
        'assets/my_1605967420_99.jpeg',
        'assets/supermarket_1605967420_70.jpeg',
        'assets/we_1605967420_78.jpeg',
        'assets/woman_1605967469_87.jpeg',
        'assets/you (plural)_1605967420_55.jpeg',
        'assets/you_1605967420_63.jpeg',
        'assets/your_1605967420_70.jpeg'
    ]
    legendas = [
        'banco', 'onibus', 'carro', 'formação', 'hospital',
        'eu', 'homem', 'motocicleta', 'Meu', 'supermercado',
        'nos', 'mulher', 'voces', 'voce', 'sua'
    ]

    colunas = st.columns(num_colunas)
    for i, (imagem_path, legenda) in enumerate(zip(imagens, legendas)):
        with colunas[i % num_colunas]:
            st.image(imagem_path, caption=legenda, width=150)


        
webrtc_streamer(
    key="hand-recognition-1", 
    video_processor_factory=VideoTransformer,
    rtc_configuration=RTC_CONFIGURATION
)

if st.button("Clique aqui para exibir as imagens"):
    exibir_imagem()
    



"""
Script para coletar imagens da câmera para treinar o modelo de reconhecimento de Libras.

Uso:
    python coletar_dados.py

Instruções:
    1. Execute o script
    2. Digite o nome do sinal (ou 'nada' para o fundo/sem sinal)
    3. Posicione sua mão na frente da câmera (ou aponte para o fundo se for 'nada')
    4. Pressione ESPAÇO para capturar uma imagem
    5. Pressione 'q' para finalizar a coleta do sinal atual
    6. Repita para cada sinal que quiser treinar
"""

import cv2
import os
from datetime import datetime
import mediapipe as mp

DATASET_DIR = "dataset"
IMAGE_SIZE = (224, 224)

def criar_estrutura_dataset():
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        print(f"✅ Pasta '{DATASET_DIR}' criada!")

def coletar_imagens_sinal(nome_sinal):
    pasta_sinal = os.path.join(DATASET_DIR, nome_sinal)
    if not os.path.exists(pasta_sinal):
        os.makedirs(pasta_sinal)
    
    cap = cv2.VideoCapture(0)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    
    if not cap.isOpened():
        print("❌ Erro: Não foi possível abrir a câmera!")
        hands.close()
        return
    
    if nome_sinal == 'nada':
        instrucao_extra = "APONTE PARA O FUNDO VAZIO (ou MÃO RELAXADA)"
        cor_instrucao = (255, 165, 0)
    else:
        instrucao_extra = "POSICIONE A MÃO(S) NO QUADRO VERDE"
        cor_instrucao = (0, 255, 0)
        
    print(f"\n📸 Coletando imagens para o sinal: '{nome_sinal}'")
    print(f"📌 Instrução: {instrucao_extra}")
    print("  - Pressione ESPAÇO para capturar uma imagem")
    print("  - Pressione 'q' para finalizar")
    print("  - Pressione 'r' para reiniciar a contagem\n")
    
    contador = 1
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Erro ao capturar frame da câmera")
            break
        
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        mao_detectada = False
        frame_cortado = None
        
        if results.multi_hand_landmarks:
            mao_detectada = True
            
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            padding = 30 
            x1 = max(0, x_min - padding)
            y1 = max(0, y_min - padding)
            x2 = min(w, x_max + padding)
            y2 = min(h, y_max + padding)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            frame_cortado = frame[y1:y2, x1:x2]
        
        texto_instrucao = f"Sinal: {nome_sinal} | Capturadas: {contador-1}"
        cv2.putText(frame, texto_instrucao, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if nome_sinal == 'nada':
            status_texto = "ESPACO para Capturar Fundo/Nada"
            cor_status = cor_instrucao
        elif mao_detectada:
            status_texto = "MAO(S) DETECTADA(S)! (ESPACO para Capturar)"
            cor_status = (0, 255, 0)
        else:
            status_texto = "POSICIONE A MAO (Q: Finalizar | R: Reiniciar)"
            cor_status = (0, 0, 255)
            
        cv2.putText(frame, status_texto, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_status, 2)
        
        cv2.imshow('Coleta de Dados - Pressione ESPACO para capturar', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            img_para_salvar = None
            
            if nome_sinal == 'nada':
                img_para_salvar = cv2.resize(frame, IMAGE_SIZE)
                cv2.putText(frame, "NADA CAPTURADO!", (w // 2 - 100, h // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 3)
            
            elif mao_detectada and frame_cortado is not None and frame_cortado.size > 0:
                img_para_salvar = cv2.resize(frame_cortado, IMAGE_SIZE)
                cv2.putText(frame, "CAPTURADA!", (w // 2 - 80, h // 2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            else:
                print("⚠️ Erro: Mão não detectada. Não é possível salvar.")
                continue
            
            if img_para_salvar is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                nome_arquivo = f"{nome_sinal}_{timestamp}_{contador:04d}.jpg"
                caminho_arquivo = os.path.join(pasta_sinal, nome_arquivo)
                
                cv2.imwrite(caminho_arquivo, img_para_salvar)
                print(f"✅ Imagem {contador} salva: {nome_arquivo}")
                contador += 1
                
                cv2.imshow('Coleta de Dados - Pressione ESPACO para capturar', frame)
                cv2.waitKey(300)
            
        elif key == ord('r'):
            contador = 1
            print("🔄 Contagem reiniciada")
            
        elif key == ord('q'):
            break
    
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    
    total_capturadas = contador - 1
    print(f"\n✅ Coleta finalizada! Total de {total_capturadas} imagens salvas em '{pasta_sinal}'")

def main():
    print("=" * 60)
    print("🤟 COLETOR DE DADOS PARA RECONHECIMENTO DE LIBRAS")
    print("=" * 60)
    
    criar_estrutura_dataset()
    
    sinais_coletados = []
    
    while True:
        print("\n" + "-" * 60)
        nome_sinal = input("Digite o nome do sinal (ou 'sair' para finalizar): ").strip()
        
        if nome_sinal.lower() == 'sair':
            break
        
        if not nome_sinal:
            print("❌ Nome do sinal não pode estar vazio!")
            continue
        
        nome_sinal_limpo = nome_sinal.replace(" ", "_").lower()
        
        coletar_imagens_sinal(nome_sinal_limpo)
        sinais_coletados.append(nome_sinal_limpo)
    
    print("\n" + "=" * 60)
    print("📊 RESUMO DA COLETA")
    print("=" * 60)
    print(f"Sinais coletados: {len(sinais_coletados)}")
    for sinal in sinais_coletados:
        pasta = os.path.join(DATASET_DIR, sinal)
        if os.path.exists(pasta):
            num_imagens = len([f for f in os.listdir(pasta) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f" - {sinal}: {num_imagens} imagens")
    print("\n✅ Coleta finalizada! Agora você pode treinar o modelo com:")
    print(f"  python treinar_modelo.py --dataset {DATASET_DIR}")

if __name__ == "__main__":
    main()
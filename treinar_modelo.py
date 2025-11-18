"""
Script para treinar um modelo de CNN para reconhecimento de sinais de Libras
usando Transfer Learning (MobileNetV2) para maior robustez.

Uso:
    python treinar_modelo.py --dataset pasta_do_dataset --output modelo.h5

Parâmetros:
    --dataset: Caminho para a pasta do dataset (padrão: 'dataset')
    --output: Nome do arquivo do modelo salvo (padrão: 'modelo_treinado.h5')
    --epochs: Número de épocas (padrão: 50)
    --batch_size: Tamanho do batch (padrão: 32)
"""

import os
import argparse
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Configurações padrão
IMAGE_WIDTH = 224 # Requerido pelo MobileNetV2
IMAGE_HEIGHT = 224 # Requerido pelo MobileNetV2
CHANNELS = 3
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.2

def carregar_dataset(dataset_dir):
    """
    Carrega imagens do dataset e retorna arrays numpy.
    
    Args:
        dataset_dir: Caminho para a pasta do dataset
        
    Returns:
        X: Array de imagens
        y: Array de labels
        label_names: Lista com nomes dos sinais
    """
    print("📂 Carregando dataset...")
    
    imagens = []
    labels = []
    label_names = []
    
    # Percorre todas as pastas no dataset
    for pasta_sinal in sorted(os.listdir(dataset_dir)):
        caminho_pasta = os.path.join(dataset_dir, pasta_sinal)
        
        # Ignora arquivos, só processa pastas
        if not os.path.isdir(caminho_pasta):
            continue
        
        label_names.append(pasta_sinal)
        label_index = len(label_names) - 1
        
        print(f"  📁 Processando: {pasta_sinal}...", end=" ")
        
        # Conta quantas imagens foram carregadas
        contador = 0
        
        # Carrega todas as imagens da pasta
        for arquivo in os.listdir(caminho_pasta):
            if arquivo.lower().endswith(('.jpg', '.jpeg', '.png')):
                caminho_imagem = os.path.join(caminho_pasta, arquivo)
                
                try:
                    # Carrega e redimensiona a imagem para o formato esperado (224x224)
                    img = cv2.imread(caminho_imagem)
                    if img is None:
                        continue
                    
                    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Converte BGR para RGB
                    
                    imagens.append(img)
                    labels.append(label_index)
                    contador += 1
                    
                except Exception as e:
                    print(f"\n  ⚠️ Erro ao carregar {arquivo}: {e}")
                    continue
        
        print(f"{contador} imagens")
    
    if len(imagens) == 0:
        raise ValueError("❌ Nenhuma imagem encontrada no dataset!")
    
    # Converte para arrays numpy
    X = np.array(imagens, dtype=np.float32)
    y = np.array(labels)
    
    # Normaliza as imagens (0-255 -> 0-1)
    X = X / 255.0
    
    print(f"\n✅ Dataset carregado:")
    print(f"  - Total de imagens: {len(X)}")
    print(f"  - Número de classes: {len(label_names)}")
    print(f"  - Classes: {', '.join(label_names)}")
    print(f"  - Formato das imagens: {X.shape[1:]} (altura x largura x canais)")
    
    return X, y, label_names

def criar_modelo_transfer_learning(num_classes, input_shape):
    """
    Cria a arquitetura do modelo usando MobileNetV2 para Transfer Learning.
    
    Args:
        num_classes: Número de classes (sinais) a classificar
        input_shape: Formato de entrada (altura, largura, canais) - Deve ser 224x224x3
        
    Returns:
        Modelo compilado do Keras
    """
    print(f"\n🏗️  Criando modelo MobileNetV2 para Transfer Learning com {num_classes} classes...")
    
    # 1. Carrega a base do MobileNetV2 (pré-treinada em milhões de imagens)
    # include_top=False: Remove a camada de classificação original
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False, 
        weights='imagenet' 
    )
    
    
    # 2. Congela os pesos da base
    # Apenas as camadas adicionadas no final serão treinadas (otimizado para dados pequenos)
    base_model.trainable = False
    
    # 3. Cria o novo modelo (camadas de topo personalizadas)
    model = keras.Sequential([
        base_model,
        # Resume as características espaciais para um vetor
        layers.GlobalAveragePooling2D(), 
        
        # Camada densa personalizada para aprendizado de alto nível
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5), # Regularização para evitar overfitting
        
        # Camada de saída para suas classes (sinais)
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # 4. Compila o modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Modelo Transfer Learning criado!")
    print("\n📊 Resumo do modelo:")
    model.summary()
    return model

# A função criar_modelo original foi removida/substituída.

def treinar_modelo(model, X_train, y_train, X_test, y_test, epochs, batch_size):
    """
    Treina o modelo.
    
    Args:
        model: Modelo do Keras
        X_train, y_train: Dados de treino
        X_test, y_test: Dados de teste
        epochs: Número de épocas
        batch_size: Tamanho do batch
        
    Returns:
        Histórico do treinamento
    """
    print(f"\n🚀 Iniciando treinamento...")
    print(f"  - Épocas: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Dados de treino: {len(X_train)}")
    print(f"  - Dados de teste: {len(X_test)}")
    
    # Callbacks para melhorar o treinamento (EarlyStopping e ReduceLROnPlateau)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Treina o modelo
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def plotar_resultados(history, output_dir):
    """
    Plota gráficos de loss e accuracy.
    
    Args:
        history: Histórico do treinamento
        output_dir: Diretório para salvar os gráficos
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Gráfico de Loss
    ax1.plot(history.history['loss'], label='Treino')
    ax1.plot(history.history['val_loss'], label='Validação')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Gráfico de Accuracy
    ax2.plot(history.history['accuracy'], label='Treino')
    ax2.plot(history.history['val_accuracy'], label='Validação')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Salva o gráfico
    grafico_path = os.path.join(output_dir, 'graficos_treinamento.png')
    plt.savefig(grafico_path)
    print(f"\n📊 Gráficos salvos em: {grafico_path}")
    plt.close()

def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description='Treina modelo de reconhecimento de Libras')
    parser.add_argument('--dataset', type=str, default='dataset',
                        help='Caminho para a pasta do dataset (padrão: dataset)')
    parser.add_argument('--output', type=str, default='modelo_treinado.h5',
                        help='Nome do arquivo do modelo (padrão: modelo_treinado.h5)')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Número de épocas (padrão: {EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help=f'Tamanho do batch (padrão: {BATCH_SIZE})')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🤟 TREINAMENTO DE MODELO DE RECONHECIMENTO DE LIBRAS (TRANSFER LEARNING)")
    print("=" * 60)
    
    # Verifica se o dataset existe
    if not os.path.exists(args.dataset):
        print(f"❌ Erro: Pasta '{args.dataset}' não encontrada!")
        print(f"  Crie a pasta e organize seus dados ou use:")
        print(f"  python coletar_dados.py")
        return
    
    # Carrega o dataset
    try:
        X, y, label_names = carregar_dataset(args.dataset)
    except Exception as e:
        print(f"❌ Erro ao carregar dataset: {e}")
        return
    
    # Converte labels para categorical
    num_classes = len(label_names)
    y_categorical = to_categorical(y, num_classes)
    
    # Divide em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    
    print(f"\n📊 Divisão dos dados:")
    print(f"  - Treino: {len(X_train)} imagens")
    print(f"  - Teste: {len(X_test)} imagens")
    
    # 🚨 CRÍTICO: Cria o modelo usando Transfer Learning
    input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS)
    model = criar_modelo_transfer_learning(num_classes, input_shape)
    
    # Treina o modelo
    history = treinar_modelo(
        model, X_train, y_train, X_test, y_test,
        args.epochs, args.batch_size
    )
    
    # Avalia o modelo
    print("\n📈 Avaliando modelo no conjunto de teste...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"  - Loss: {test_loss:.4f}")
    print(f"  - Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Salva o modelo
    output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else '.'
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model.save(args.output)
    print(f"\n💾 Modelo salvo em: {args.output}")
    
    # Salva informações sobre o modelo
    info_path = args.output.replace('.h5', '_info.txt')
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("INFORMAÇÕES DO MODELO TREINADO\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Número de classes: {num_classes}\n")
        f.write(f"Classes: {', '.join(label_names)}\n\n")
        f.write(f"Accuracy no teste: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n")
        f.write(f"Loss no teste: {test_loss:.4f}\n\n")
        f.write("Mapeamento de classes:\n")
        for i, nome in enumerate(label_names):
            f.write(f" {i}: {nome}\n")
    
    print(f"📄 Informações salvas em: {info_path}")
    
    # Plota gráficos
    plotar_resultados(history, output_dir)
    
    # Salva o dicionário de labels para usar no app
    labels_path = args.output.replace('.h5', '_labels.py')
    with open(labels_path, 'w', encoding='utf-8') as f:
        f.write("# Dicionário de labels para usar no app.py\n")
        f.write("label_to_text = {\n")
        for i, nome in enumerate(label_names):
            f.write(f" {i}: '{nome}',\n")
        f.write("}\n")
    
    print(f"📝 Dicionário de labels salvo em: {labels_path}")
    
    print("\n" + "=" * 60)
    print("✅ TREINAMENTO CONCLUÍDO!")
    print("=" * 60)
    print(f"\n📋 Próximos passos:")
    print(f"  1. **Mova** o arquivo `{args.output}` para a pasta do seu app.")
    print(f"  2. **Mova** o arquivo `{labels_path}` (ou renomeie-o para `modelo_treinado_labels.py`) para a pasta do seu app.")
    print(f"  3. Execute o app:")
    print(f"   streamlit run app/app.py")

if __name__ == "__main__":
    main()
# Reconhecimento de Sinais em Libras

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![License](https://img.shields.io/badge/License-MIT-green)

Aplicação web para reconhecimento de sinais da Língua Brasileira de Sinais (Libras) em tempo real usando Deep Learning e visão computacional.

[Recursos](#recursos) • [Instalação](#instalação) • [Uso](#uso) • [Treinamento](#treinamento-de-modelo) • [Coleta de Dados](#coleta-de-dados) • [Estrutura](#estrutura-do-projeto)

</div>

---

## Sobre o Projeto

Este projeto utiliza redes neurais convolucionais (CNN) com Transfer Learning (MobileNetV2) para reconhecer sinais de Libras através da webcam em tempo real. O sistema combina:

- **MediaPipe** para detecção de mãos
- **TensorFlow/Keras** para classificação de sinais
- **Streamlit** para interface web interativa
- **streamlit-webrtc** para processamento de vídeo em tempo real

## Recursos

- Reconhecimento em tempo real via webcam
- Transfer Learning com MobileNetV2 para maior precisão
- Detecção de múltiplas mãos simultaneamente
- Interface interativa com ajuste de threshold de confiança
- Modo Debug para análise de predições
- Galeria de exemplos de sinais coletados
- Sistema completo de coleta e treinamento de dados

## Tecnologias

- **Python 3.8+**
- **TensorFlow 2.x** - Framework de Deep Learning
- **MediaPipe** - Detecção de mãos
- **OpenCV** - Processamento de imagens
- **Streamlit** - Interface web
- **streamlit-webrtc** - Streaming de vídeo
- **NumPy** - Computação numérica
- **scikit-learn** - Pré-processamento de dados

## Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)
- Webcam funcional
- (Opcional) FFmpeg para melhor compatibilidade de vídeo

### Passo a passo

1. **Clone o repositório**
   ```bash
   git clone https://github.com/seu-usuario/Libra-transcription.git
   cd Libra-transcription
   ```

2. **Crie um ambiente virtual**
   
   **Windows:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
   
   **Linux/Mac:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute a aplicação**
   ```bash
   streamlit run app/app.py
   ```

5. **Acesse no navegador**
   ```
   http://localhost:8501
   ```

## Uso

### Executando a Aplicação

1. Inicie o app com `streamlit run app/app.py`
2. Permita o acesso à webcam quando solicitado
3. Posicione sua mão na frente da câmera
4. O sistema detectará e classificará o sinal em tempo real
5. Ajuste o threshold de confiança na barra lateral conforme necessário

### Configurações Disponíveis

- **Threshold de Confiança** (0.0 - 1.0): Define o nível mínimo de confiança para exibir predições
- **Modo Debug**: Exibe as top 3 predições no console para análise
- **Galeria de Exemplos**: Visualize os sinais disponíveis no dataset

## Treinamento de Modelo

### Preparar Dataset

1. Organize as imagens em pastas por classe:
   ```
   dataset/
   ├── sinal_1/
   │   ├── imagem1.jpg
   │   ├── imagem2.jpg
   │   └── ...
   ├── sinal_2/
   │   └── ...
   └── ...
   ```

### Treinar Modelo

```bash
python treinar_modelo.py --dataset dataset --output modelo_novo.h5 --epochs 50 --batch_size 32
```

**Parâmetros:**
- `--dataset`: Caminho para a pasta do dataset (padrão: `dataset`)
- `--output`: Nome do arquivo do modelo (padrão: `modelo_treinado.h5`)
- `--epochs`: Número de épocas de treinamento (padrão: `50`)
- `--batch_size`: Tamanho do batch (padrão: `32`)

### Arquivos Gerados

Após o treinamento, serão criados:
- `modelo_novo.h5` - Modelo treinado
- `modelo_novo_labels.py` - Dicionário de classes
- `modelo_novo_info.txt` - Informações e métricas
- `graficos_treinamento.png` - Gráficos de loss e accuracy

## Coleta de Dados

### Coletar Imagens

Execute o script de coleta:

```bash
python coletar_dados.py
```

**Instruções durante a coleta:**
1. Digite o nome do sinal quando solicitado
2. Posicione sua mão na frente da câmera
3. Pressione **ESPAÇO** para capturar uma imagem
4. Pressione **Q** para finalizar a coleta do sinal atual
5. Pressione **R** para reiniciar a contagem
6. Digite `sair` para encerrar o programa

**Dicas para melhor coleta:**
- Capture entre 50-200 imagens por sinal
- Varie o ângulo e posição da mão
- Use diferentes iluminações
- Inclua diferentes tons de pele
- Mantenha fundo variado

## Estrutura do Projeto

```
Libra-transcription/
├── app/
│   └── app.py                      # Aplicação Streamlit principal
├── dataset/                        # Imagens organizadas por classe
│   ├── classe_1/
│   ├── classe_2/
│   └── ...
├── coletar_dados.py               # Script de coleta de imagens
├── treinar_modelo.py              # Script de treinamento
├── modelo_treinado.h5             # Modelo principal
├── modelo_treinado_labels.py      # Mapeamento de classes
├── requirements.txt               # Dependências do projeto
└── README.md                      # Este arquivo
```

## Componentes Principais

### `app/app.py`
- **`predict_object()`**: Realiza a predição do sinal detectado
- **`VideoTransformer`**: Classe para processamento de vídeo em tempo real
- **`show_sign_examples()`**: Exibe galeria de exemplos de sinais

### `treinar_modelo.py`
- **`carregar_dataset()`**: Carrega e pré-processa imagens
- **`criar_modelo_transfer_learning()`**: Cria modelo com MobileNetV2
- **`treinar_modelo()`**: Executa o treinamento com callbacks
- **`plotar_resultados()`**: Gera gráficos de desempenho

### `coletar_dados.py`
- **`criar_estrutura_dataset()`**: Cria estrutura de pastas
- **`coletar_imagens_sinal()`**: Interface de coleta com MediaPipe

## Modelos Disponíveis

O projeto inclui diferentes modelos treinados:

| Modelo | Classes | Accuracy | Descrição |
|--------|---------|----------|-----------|
| `modelo_treinado.h5` | 15 | ~85% | Modelo principal com 15 sinais |
| `modelo_final_9classes.h5` | 9 | ~90% | Modelo focado em 9 sinais básicos |
| `modelo_tl_final.h5` | Variável | ~88% | Transfer Learning aprimorado |

## Configurações Avançadas

### Ajustar Threshold de Confiança

No arquivo `app/app.py`:
```python
CONFIDENCE_THRESHOLD = 0.25  # Ajuste entre 0.0 e 1.0
```

### Modificar Parâmetros de Treinamento

No arquivo `treinar_modelo.py`:
```python
EPOCHS = 50           # Número de épocas
BATCH_SIZE = 32       # Tamanho do batch
LEARNING_RATE = 0.001 # Taxa de aprendizado
TEST_SIZE = 0.2       # Percentual de teste (20%)
```

### Configurar MediaPipe

No arquivo `app/app.py`:
```python
self.hands = mp_hands.Hands(
    max_num_hands=2,              # Máximo de mãos detectadas
    min_detection_confidence=0.5,  # Confiança mínima de detecção
    min_tracking_confidence=0.5    # Confiança mínima de rastreamento
)
```

## Solução de Problemas

### Erro ao abrir a câmera
- Verifique se nenhum outro aplicativo está usando a webcam
- Tente alterar o índice da câmera em `cv2.VideoCapture(0)` para `1` ou `2`

### Erro com streamlit-webrtc
- Instale FFmpeg no sistema
- Windows: `choco install ffmpeg`
- Linux: `sudo apt-get install ffmpeg`
- Mac: `brew install ffmpeg`

### Baixa precisão do modelo
- Colete mais imagens (mínimo 100 por classe)
- Aumente o número de épocas de treinamento
- Varie as condições de iluminação e fundo
- Use data augmentation durante o treinamento

### Modelo não encontrado
- Verifique se o arquivo `.h5` está na raiz do projeto
- Certifique-se de que o arquivo `*_labels.py` também está presente
- Ajuste o caminho em `app/app.py` se necessário

## Métricas de Desempenho

O sistema gera automaticamente:
- **Gráficos de Loss**: Treino vs Validação
- **Gráficos de Accuracy**: Treino vs Validação
- **Matriz de Confusão**: (implementar se necessário)
- **Relatório de Classificação**: Precision, Recall, F1-Score

## Contribuindo

Contribuições são bem-vindas! Para contribuir:

1. Faça um Fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/NovaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/NovaFeature`)
5. Abra um Pull Request


---

<div align="center">

**Se este projeto foi útil, considere dar uma estrela!**

</div>

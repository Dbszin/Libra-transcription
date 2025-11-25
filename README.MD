# Reconhecimento de Sinais (Libras)

Resumo
- App Streamlit para reconhecer sinais (Libras) via webcam usando MediaPipe + TensorFlow + streamlit-webrtc.
- Código principal: [app/app.py](app/app.py) com as funções/classes [`predict_object`](app/app.py) e [`VideoTransformer`](app/app.py).

Requisitos
- Instale as dependências listadas em [requirements.txt](requirements.txt):
  - Ex.: streamlit, tensorflow, mediapipe, opencv-python, streamlit-webrtc, av, etc.

Estrutura do projeto
- [app/app.py](app/app.py) — app Streamlit e processamento de vídeo em tempo real. Contém [`predict_object`](app/app.py) e [`VideoTransformer`](app/app.py).
- [treinar_modelo.py](treinar_modelo.py) — script de treino (salva .h5 e _labels.py).
- [coletar_dados.py](coletar_dados.py) — script para coletar imagens via webcam.
- Modelos:
  - [modelo_treinado.h5](modelo_treinado.h5)
  - [modelo_treinado_labels.py](modelo_treinado_labels.py)
  - [modelo_final_9classes.h5](modelo_final_9classes.h5)
  - [modelo_tl_final.h5](modelo_tl_final.h5)
- [dataset/](dataset/) — imagens por classe.
- [requirements.txt](requirements.txt)

Como rodar (local)
1. Criar e ativar ambiente virtual:
   - Windows: .venv\Scripts\activate
   - Linux/Mac: source .venv/bin/activate
2. Instalar dependências:
   pip install -r requirements.txt
3. Executar app:
   streamlit run app/app.py

Observações importantes
- PyAV (`av`) e streamlit-webrtc podem requerer ffmpeg e bibliotecas do sistema.
- O app carrega por padrão [modelo_treinado.h5](modelo_treinado.h5). Se trocar de modelo, mova o .h5 e o arquivo de labels (ex.: `modelo_treinado_labels.py`) para a raiz do projeto ou ajuste os caminhos em [app/app.py](app/app.py).
- Ajuste o threshold de confiança pela sidebar do app ou pela constante `CONFIDENCE_THRESHOLD` em [app/app.py](app/app.py).

Como gerar novo modelo
- Use [`treinar_modelo.py`](treinar_modelo.py). Ele salva o arquivo .h5 e um arquivo de labels (`*_labels.py`) — mover esses arquivos para a raiz para o app encontrá-los.

Coletar dados
- Use [`coletar_dados.py`](coletar_dados.py) para popular [dataset/](dataset/) com imagens por classe.

Links rápidos
- Código do app: [app/app.py](app/app.py)
- Funções-chave: [`predict_object`](app/app.py), [`VideoTransformer`](app/app.py)
- Treino: [treinar_modelo.py](treinar_modelo.py)
- Coleta: [coletar_dados.py](coletar_dados.py)
- Models/labels: [modelo_treinado.h5](modelo_treinado.h5), [modelo_treinado_labels.py](modelo_treinado_labels.py)
- Dependências: [requirements.txt](requirements.txt)

# Painel do laboratorio. Constroi a partir da RAIZ do repositorio:
#
#     docker build -f server/Dockerfile -t xbd-lab .
#
# No Coolify: Build Pack = Dockerfile, Dockerfile Location = server/Dockerfile,
# Base Directory = / (a raiz, para que o COPY de experiments/ funcione).
#
# Sem torch aqui de proposito. O servidor so recebe numeros por HTTP e os escreve
# como tfevents; puxar CUDA para dentro da imagem custaria alguns GB por nada.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    LAB_DATA=/data

WORKDIR /app

COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server/app ./app
# A matriz entra na imagem para o primeiro deploy ja subir com a fila montada.
# Um `POST /api/matriz` grava em /data/matrix.yaml, que tem precedencia e
# sobrevive ao proximo deploy.
COPY experiments/matrix.yaml /app/matrix.yaml

RUN mkdir -p /data && useradd -u 10001 -m lab && chown -R lab /data /app
USER lab
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request;\
urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/api/saude',timeout=4)"

# Um worker so: o Store e um SQLite com um escritor, e o TensorBoard sobe como
# subprocesso no lifespan. Dois workers dariam dois TensorBoard disputando a
# mesma porta. A carga e de 20 maquinas mandando uma requisicao a cada 5 s.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]

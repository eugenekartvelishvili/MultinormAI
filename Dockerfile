FROM nvidia/cuda:13.0.2-base-ubuntu22.04

COPY usergate-root.crt /usr/local/share/ca-certificates/usergate-root.crt
RUN update-ca-certificates

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    APP_DIR=/app \
    MODELS_DIR=/app/models \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt

WORKDIR $APP_DIR
USER root

# Системные зависимости
RUN apt-get update && apt-get install -y \
    wget curl ca-certificates bzip2 git build-essential \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Python 3.12 через Miniconda
RUN curl -fsSL --insecure \
    https://repo.anaconda.com/miniconda/Miniconda3-py312_24.7.1-0-Linux-x86_64.sh \
    -o /tmp/miniconda.sh \
 && bash /tmp/miniconda.sh -b -p /opt/conda \
 && rm /tmp/miniconda.sh \
 && /opt/conda/bin/conda clean -afy

ENV PATH=/opt/conda/bin:$PATH

# Папка для моделей
RUN mkdir -p $MODELS_DIR && chmod -R 777 $MODELS_DIR

# Оффлайн wheels
COPY wheels /wheels
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-index --find-links=/wheels torch torchvision 2>/dev/null || true

# Основные зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код
COPY ./app/code /app/code

CMD ["bash"]
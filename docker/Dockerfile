FROM python:3.11-slim

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt scripts/install_lightfm.sh /workspace/
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && PYTHON=python bash /workspace/install_lightfm.sh

COPY . /workspace/recsys_training
RUN pip install -e /workspace/recsys_training

EXPOSE 8888
CMD ["jupyter", "lab", "--no-browser", "--ip=0.0.0.0", "--port=8888", \
     "--allow-root", "--notebook-dir=/workspace/recsys_training/notebooks", \
     "--NotebookApp.token="]

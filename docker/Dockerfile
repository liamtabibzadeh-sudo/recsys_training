FROM continuumio/miniconda3:latest

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

WORKDIR /workspace
COPY environment.yaml /workspace/environment.yaml

RUN conda env create -f /workspace/environment.yaml

COPY . /workspace/recsys_training
RUN /bin/bash -c "source activate recsys_training && pip install -e /workspace/recsys_training"

EXPOSE 8888
CMD ["/bin/bash", "-lc", "source activate recsys_training && jupyter lab --no-browser --ip=0.0.0.0 --port=8888 --allow-root --notebook-dir=/workspace/recsys_training/notebooks --NotebookApp.token=''"]

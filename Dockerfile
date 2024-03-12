ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=11.8.0

FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION}

SHELL ["/bin/bash", "-c"]

RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV CONDA_HOME /opt/conda
ENV PATH $CONDA_HOME/bin:$PATH
ENV CUDA_HOME $CONDA_HOME
ENV KMP_INIT_AT_FORK FALSE

RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    bash Miniforge3-$(uname)-$(uname -m).sh -b -p $CONDA_HOME && \
    rm Miniforge3-$(uname)-$(uname -m).sh

COPY environment.yml ./

RUN mamba env update -n base -f environment.yml && \
    conda clean --all && \
    pip cache purge

WORKDIR /mnt

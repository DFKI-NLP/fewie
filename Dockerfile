FROM nvcr.io/nvidia/pytorch:21.02-py3

LABEL maintainer="leonhard.hennig@dfki.de"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install base packages.
RUN apt-get update --fix-missing && apt-get install -y \
    supervisor \
    bzip2 \
    ca-certificates \
    curl \
    gcc \
    git \
    libc-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    libevent-dev \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app/fewie
#COPY gpu-requirements.txt .

# Install packages and skip some that are already included
COPY * ./
RUN pip3 install .

VOLUME ["/app/fewie/"]


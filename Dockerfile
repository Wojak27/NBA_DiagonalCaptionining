# Base image with CUDA support
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu18.04

# Set environment variables
# ENV DEBIAN_FRONTEND=noninteractive

# Install Java 1.8 and Python 3.6.9
RUN apt-get update && apt-get install -y \
    openjdk-8-jdk \
    python3.7 \
    git \
    python3-pip \
    python3.7-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.7 as the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

RUN python3 -m pip install --upgrade pip

RUN python3  -m pip  install --pre torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
# Install pip packages
RUN python3 -m pip install \
    tensorboard \
    wandb \
    torchsummary \
    einops \
    pickle5 \
    pyrsistent \
    boto3 \
    tqdm \
    requests \ 
    pandas

RUN python3 -m pip install git+https://github.com/Maluuba/nlg-eval.git@master

RUN nlg-eval --setup


# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# make it execute as a python script
ENTRYPOINT ["python3", "main_task_caption_DAM.py"]
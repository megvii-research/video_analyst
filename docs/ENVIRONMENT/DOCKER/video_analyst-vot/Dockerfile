# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.6    (apt)
# pytorch       latest (pip)
# ==================================================================
# FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
ENV LANG C.UTF-8

ENV APT_INSTALL="apt-get install -y --no-install-recommends" \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple --default-timeout=100000" \
    GIT_CLONE="git clone --depth 1"

# added by user
COPY ./sources.list.tuna.txt /etc/apt/
RUN bash -xc " \
  echo 'change sources.list to tuna source' && \
  pushd /etc/apt && \
  mv sources.list sources.list.origin.txt && \
  mv sources.list.tuna.txt sources.list && \
  popd \
"

RUN rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \

    apt-get update && \

# ==================================================================
# tools
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        git \
        vim \
        libssl-dev \
        curl \
        unzip \
        unrar \
        && \

    $GIT_CLONE https://github.com/Kitware/CMake ~/cmake && \
    cd ~/cmake && \
    ./bootstrap && \
    make -j"$(nproc)" install && \

# ==================================================================
# python
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.6 \
        python3.6-dev \
        python3-distutils-extra \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.6 ~/get-pip.py && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.6 /usr/local/bin/python && \
    $PIP_INSTALL \
        setuptools \
        && \
    $PIP_INSTALL \
        # numpy \
        # scipy \
        # pandas \
        cloudpickle \
        # scikit-image>=0.14.2 \
        # scikit-learn \
        # matplotlib \
        # Cython \
        tqdm \
        && \

# ==================================================================
# pytorch
# ------------------------------------------------------------------

    $PIP_INSTALL \
        future \
        # numpy \
        protobuf \
        enum34 \
        pyyaml \
        typing \
        && \
    
#    $PIP_INSTALL \
#        --pre torch torchvision -f \
#        https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html \
#        && \

#    $PIP_INSTALL torch==1.5.1 torchvision==0.6.1 && \ 

# ==================================================================
# config & cleanup
# ------------------------------------------------------------------

    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*

# ==================================================================
# additional apt pacakges
# ------------------------------------------------------------------

# personal tmux config file
COPY ./.tmux.conf /root/
RUN apt update && \
        $APT_INSTALL tmux && \
        # needed by opencv
        $APT_INSTALL libsm6 libxrender1 libxext6 && \ 
        $APT_INSTALL htop && \ 
    echo "tmux installed."

# TODO: add cloning of https://github.com/tmux-plugins/tmux-resurrect @ /root/Utils

# ==================================================================
# video_analyst
# ------------------------------------------------------------------
COPY ./requirements.txt /root/ 
RUN python setup.py develop && \
    echo "ended."

# TODO: only wget & install: https://github.com/MARMOTatZJU/video_analyst/blob/dev/requirements.txt

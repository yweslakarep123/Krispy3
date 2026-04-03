FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MUJOCO_GL=egl
ENV D4RL_SUPPRESS_IMPORT_ERROR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget curl git build-essential gcc g++ \
    libx11-dev libgl1-mesa-dev libegl1-mesa-dev libglew-dev \
    libosmesa6-dev libglu1-mesa-dev \
    patchelf unzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/.mujoco && \
    wget -q https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz \
      -O /tmp/mujoco210.tar.gz --no-check-certificate && \
    tar -xzf /tmp/mujoco210.tar.gz -C /root/.mujoco/ && \
    rm /tmp/mujoco210.tar.gz

ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH}

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
      -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:${PATH}

RUN conda create -n flowpolicy python=3.8 -y
SHELL ["conda", "run", "-n", "flowpolicy", "/bin/bash", "-c"]

RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu117

RUN pip install setuptools==59.5.0 Cython==0.29.35 patchelf==0.17.2.0

WORKDIR /workspace
COPY . /workspace/FlowPolicy

RUN cd /workspace/FlowPolicy/third_party/mujoco-py-2.1.2.14 && pip install -e .

RUN cd /workspace/FlowPolicy/third_party/gym-0.21.0 && pip install -e . && \
    cd /workspace/FlowPolicy/third_party/Metaworld && pip install -e . && \
    cd /workspace/FlowPolicy/third_party/rrl-dependencies && \
    pip install -e mj_envs/. && pip install -e mjrl/.

RUN cd /workspace/FlowPolicy/third_party/pytorch3d_simplified && pip install -e .

RUN pip install \
    zarr==2.12.0 wandb ipdb gpustat \
    "mujoco<3.0" "dm_control<1.0.15" \
    omegaconf hydra-core==1.2.0 \
    dill==0.3.5.1 einops==0.4.1 diffusers==0.11.1 "huggingface_hub<0.24" \
    numba==0.56.4 moviepy imageio av matplotlib termcolor natsort open3d

RUN pip install git+https://github.com/Farama-Foundation/D4RL@master#egg=d4rl

RUN cd /workspace/FlowPolicy/FlowPolicy && pip install -e .

RUN conda run -n flowpolicy python -c "import mujoco_py" 2>/dev/null || true

WORKDIR /workspace/FlowPolicy

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "flowpolicy"]
CMD ["bash"]

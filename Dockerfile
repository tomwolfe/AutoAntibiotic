FROM ubuntu:22.04

LABEL maintainer="AutoAntibiotic Team"
LABEL description="AutoAntibiotic Discovery Pipeline — MRSA PBP2a Inhibitor Screening"

SHELL ["/bin/bash", "-c"]

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# ── System dependencies ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    git \
    ca-certificates \
    libxrender1 \
    libxext6 \
    libsm6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# ── Miniconda ───────────────────────────────────────────────────────
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniconda.sh && \
    conda config --set always_yes yes && \
    conda config --add channels conda-forge && \
    conda update -n base conda

# ── RDKit, OpenBabel, AutoDock Vina ─────────────────────────────────
RUN conda install -c conda-forge \
    python=3.11 \
    rdkit==2023.9.6 \
    openbabel==3.1.1 \
    vina==1.2.3 \
    numpy==1.26.2 \
    pandas==2.1.5 \
    matplotlib==3.8.3 \
    biopython==1.83 \
    pyyaml \
    plotly>=5.18.0 \
    scikit-learn>=1.3.0 \
    && conda clean --all -f -y

# ── Python packages (pip) ───────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# ── Application code ────────────────────────────────────────────────
WORKDIR /app
COPY . .

# Entry point
ENTRYPOINT ["python", "-m", "autoantibiotic"]
CMD ["--help"]

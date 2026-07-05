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

# ── ADFR Suite (prepare_receptor) ───────────────────────────────────
RUN wget --quiet https://ccsb.scripps.edu/adfr/downloads/adfr-1.0rc1-Linux-64bit.tar.gz \
    -O /tmp/adfr.tar.gz && \
    mkdir -p /opt/adfr && \
    tar xzf /tmp/adfr.tar.gz -C /opt/adfr --strip-components=1 && \
    rm /tmp/adfr.tar.gz && \
    ln -s /opt/adfr/bin/prepare_receptor /usr/local/bin/prepare_receptor && \
    chmod +x /usr/local/bin/prepare_receptor && \
    prepare_receptor --help > /dev/null 2>&1 && \
    echo "ADFR Suite installed: prepare_receptor OK"

# ── GNINA (pre-compiled binary, CPU-only) ───────────────────────────
RUN wget --quiet https://github.com/gnina/gnina/releases/latest/download/gnina \
    -O /usr/local/bin/gnina && \
    chmod +x /usr/local/bin/gnina && \
    gnina --help > /dev/null 2>&1 && \
    echo "GNINA installed: OK"

# ── Verify all binaries ─────────────────────────────────────────────
RUN echo "=== Binary Verification ===" && \
    for bin in vina gnina obabel prepare_receptor; do \
        if command -v $bin &> /dev/null; then \
            echo "  ✓ $bin found at $(which $bin)"; \
        else \
            echo "  ✗ $bin NOT FOUND"; \
            exit 1; \
        fi; \
    done && \
    echo "=== All binaries verified ==="

# ── Python packages (pip) ───────────────────────────────────────────
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# ── Startup verification script ─────────────────────────────────────
COPY docker/healthcheck.sh /healthcheck.sh 2>/dev/null || true
RUN if [ ! -f /healthcheck.sh ]; then \
        echo '#!/bin/bash' > /healthcheck.sh && \
        echo 'for bin in vina gnina obabel prepare_receptor; do' >> /healthcheck.sh && \
        echo '  if ! command -v $bin &> /dev/null; then' >> /healthcheck.sh && \
        echo '    echo "MISSING: $bin"' >> /healthcheck.sh && \
        echo '    exit 1' >> /healthcheck.sh && \
        echo '  fi' >> /healthcheck.sh && \
        echo 'done' >> /healthcheck.sh && \
        echo 'python -c "from autoantibiotic import CONFIG; print(\"Python imports OK\")"' >> /healthcheck.sh && \
        echo 'echo "All dependencies available"' >> /healthcheck.sh; \
    fi && \
    chmod +x /healthcheck.sh

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD /healthcheck.sh

# ── Application code ────────────────────────────────────────────────
WORKDIR /app
COPY . .

# Entry point
ENTRYPOINT ["python", "-m", "autoantibiotic"]
CMD ["--help"]

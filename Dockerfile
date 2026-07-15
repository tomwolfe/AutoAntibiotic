# AutoAntibiotic Discovery Pipeline — container image
#
# Provides a zero-install environment: AutoDock Vina and OpenBabel are installed
# via conda-forge, and the Python package (with its CLI `autoantibiotic`) is
# installed from pyproject.toml. A user can screen a compound without any local
# setup:
#
#   docker build -t autoantibiotic .
#   docker run -v $(pwd)/output:/app/output autoantibiotic --smiles "CC1C2C(...)"
#
FROM continuumio/miniconda3:latest

# Avoid interactive prompts during conda operations.
ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_PYTHON=/opt/conda/bin/python

WORKDIR /app

# Copy only what is needed to resolve and install Python dependencies first so
# this layer is cached unless pyproject.toml changes.
COPY pyproject.toml ./

# Create the dedicated environment and install the external binaries that are
# not pip-installable (vina, openbabel) plus the package itself.
RUN conda create -y -n autoantibiotic python=3.10 && \
    conda install -y -n autoantibiotic -c conda-forge vina openbabel && \
    /opt/conda/envs/autoantibiotic/bin/pip install --no-cache-dir .[docking] && \
    conda clean -afy

# Activate the environment for every subsequent RUN / CMD / ENTRYPOINT.
ENV PATH=/opt/conda/envs/autoantibiotic/bin:$PATH

# Bring in the rest of the repository (kept in its own layer for faster rebuilds
# during development).
COPY . .

# The output directory is where all reports/artifacts land. It is created so it
# can be mounted easily with `-v $(pwd)/output:/app/output`.
RUN mkdir -p /app/output

ENTRYPOINT ["autoantibiotic"]
CMD ["--help"]

# VLMShift in a container: the whole tool on your machine, with no account anywhere.
#
#   docker build -t vlmshift .                               the tool, with the fake models of the demo
#   docker build -t vlmshift --build-arg EXTRAS=local .      with torch and transformers, for real models
#   docker run --rm -v "$PWD:/work" vlmshift all experiment.yaml analysis.yaml
#
# Your folder is mounted on /work: the table, the images and the workspace stay on your disk, and
# nothing is sent anywhere. A real model is downloaded from Hugging Face at its first use; mount a
# cache (-v ~/.cache/huggingface:/root/.cache/huggingface) and set HF_HUB_OFFLINE=1 to stay offline.
FROM python:3.11-slim

WORKDIR /opt/vlmshift
COPY pyproject.toml README.md ./
COPY src ./src
ARG EXTRAS=""
RUN pip install --no-cache-dir ".${EXTRAS:+[$EXTRAS]}"

WORKDIR /work
ENTRYPOINT ["vlmshift"]

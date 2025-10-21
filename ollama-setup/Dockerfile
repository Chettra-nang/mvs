FROM python:3.12-slim

# System deps (add more if your repo needs them)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Do not COPY repository files at build time. Instead install requirements at container start
# so the build succeeds even when requirements.txt is only available at runtime via mount.
RUN pip install --upgrade pip

# Add an entrypoint that installs requirements if present in the mounted workspace
COPY test2/ollama-setup/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set environment so Ollama client uses the container host name for the Ollama service
ENV OLLAMA_API_URL=http://ollama:11434

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]

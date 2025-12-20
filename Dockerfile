FROM python:3.13-slim-bookworm

WORKDIR /app

# Install system dependencies (git required for requirements.txt)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt .
COPY pyproject.toml .
COPY MANIFEST.in .
COPY README.md .
COPY LICENSE.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and other directories
COPY ./ctinexus ./ctinexus
COPY ./mitre-ttp-mapping-main ./mitre-ttp-mapping-main
COPY ./lib ./lib
COPY ./tests ./tests

# Install ctinexus package in editable mode
RUN pip install -e .

EXPOSE 8000

EXPOSE 57623-57628

CMD ["python3", "-m", "ctinexus.app"]

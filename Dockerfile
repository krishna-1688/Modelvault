FROM python:3.11-slim AS base

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home --uid 1000 modelvault && chown -R modelvault:modelvault /app
USER modelvault

EXPOSE 8000 8501

# Default command runs the gateway; docker-compose overrides this for the
# dashboard service. Model artifacts (artifacts/) and the dataset cache
# (data/raw/) are expected to be mounted as volumes -- training is a
# separate, explicit step (`docker compose run gateway python -m modelvault.model.train`),
# not baked into the image build.
CMD ["uvicorn", "modelvault.gateway.api:app", "--host", "0.0.0.0", "--port", "8000"]

# VisionGuard: AI vs Real Image Classification

[![CI](https://github.com/Munam32/23L-3019_A2_Ops/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Munam32/23L-3019_A2_Ops/actions/workflows/ci.yml)

A Flask web service that tells whether an uploaded image is AI-generated or real. A Keras model
(EfficientNetB0, fine-tuned) is tracked with MLflow on DagsHub; the app logs every prediction to a database
and runs as a Docker Compose stack.

## Run locally with Docker

```bash
cp .env.example .env          # then set POSTGRES_PASSWORD
docker compose up -d --build
```

Open http://localhost:5000. The champion model is read from `./models/champion.keras` (mounted read-only).

## Project links

- DagsHub: https://dagshub.com/Munam32/23L-3019_A2_Ops

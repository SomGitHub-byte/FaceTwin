#!/usr/bin/env bash
# FaceTwin — Build and deploy script (Local Docker or GCP Cloud Run)
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-facetwin}"
CONTAINER_NAME="${CONTAINER_NAME:-facetwin-app}"
PORT="${GRADIO_SERVER_PORT:-7860}"
MODE="local"

# Parse arguments
for arg in "$@"; do
  case $arg in
    --cloud|--cloud-run)
      MODE="cloud"
      shift
      ;;
    *)
      # Ignore other arguments
      ;;
  esac
done

if [ "$MODE" = "cloud" ]; then
  echo "==> Deploying to Google Cloud Run..."
  echo "Make sure you have run 'gcloud config set project YOUR_PROJECT_ID'"
  echo "and registered a secret named 'GEMINI_API_KEY' in Secret Manager (if using online mode)."
  
  # Deploy to Cloud Run using gcloud CLI.
  # --source . uploads code and builds the container via Cloud Build using the Dockerfile.
  gcloud run deploy "${CONTAINER_NAME}" \
    --source . \
    --port 7860 \
    --allow-unauthenticated \
    --set-env-vars=NARRATOR_MODE=gemini,RATE_LIMIT_PER_MINUTE=10 \
    --update-secrets=GEMINI_API_KEY=GEMINI_API_KEY:latest
  
  echo "==> Cloud Run deployment complete!"
else
  echo "==> Building Local Docker image: ${IMAGE_NAME}"
  docker build -t "${IMAGE_NAME}:latest" .

  echo "==> Stopping existing container (if any)"
  docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

  echo "==> Starting FaceTwin on http://localhost:${PORT} (offline template mode)"
  docker run -d \
    --name "${CONTAINER_NAME}" \
    -p "${PORT}:7860" \
    -e GRADIO_SERVER_PORT=7860 \
    -e NARRATOR_MODE="${NARRATOR_MODE:-template}" \
    -e RATE_LIMIT_PER_MINUTE="${RATE_LIMIT_PER_MINUTE:-10}" \
    "${IMAGE_NAME}:latest"

  echo "==> Done. Logs: docker logs -f ${CONTAINER_NAME}"
fi

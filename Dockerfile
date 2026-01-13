# Combined Frontend + Backend Dockerfile for Render
# Stage 1: Build Frontend
FROM node:20-alpine AS frontend-builder

WORKDIR /frontend

# Copy frontend files
COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./

# Build Next.js static export
ENV NEXT_PUBLIC_API_URL=""
ENV NEXT_PUBLIC_WS_URL=""
RUN npm run build

# Stage 2: Backend with Frontend static files
FROM python:3.11-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./

# Copy frontend build output to static directory
COPY --from=frontend-builder /frontend/out ./static

# Create logs directory
RUN mkdir -p logs

# Expose port (Render sets PORT env)
EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-10000}/api/status || exit 1

# Run the API server (serves both API and frontend)
CMD ["sh", "-c", "uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-10000}"]

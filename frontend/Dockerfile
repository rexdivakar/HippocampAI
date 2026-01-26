# Simple Dockerfile using Vite dev server with proxy support

FROM node:20-alpine

WORKDIR /app

# Set environment variable for API URL (Docker backend service name)
ENV VITE_API_URL=http://hippocampai:8000

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Expose port 81
EXPOSE 81

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD wget --quiet --tries=1 --spider http://localhost:81/ || exit 1

# Start Vite dev server (has proxy support)
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "81"]

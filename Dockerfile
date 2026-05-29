# Stage 1: Build the React frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app
COPY package*.json ./
COPY bun.lock* ./
RUN npm install
COPY . .
RUN npm run build

# Stage 2: Run the FastAPI backend and serve the app
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies needed for OpenCV and PyTorch
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create user with UID 1000
RUN useradd -m -u 1000 user

# Copy backend requirements first to leverage Docker caching
COPY backend/requirements.txt ./backend-requirements.txt
COPY backend/forensic_ai/requirements.txt ./forensic-requirements.txt

# Install python packages
RUN pip install --no-cache-dir -r backend-requirements.txt
RUN pip install --no-cache-dir -r forensic-requirements.txt

# Create data folder and set permissions
RUN mkdir -p /data && chown -R user:user /data && chmod 777 /data

# Copy backend application source code (including forensic_ai)
COPY backend/ /app/backend/
# Copy the built React static files to backend/static
COPY --from=frontend-builder /app/dist /app/backend/static

# Set ownership of app directory to user
RUN chown -R user:user /app

# Switch to non-root user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory to backend folder
WORKDIR /app/backend

# Expose default Hugging Face Spaces port
EXPOSE 7860

# Run FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

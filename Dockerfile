# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy your Python script into the container
COPY NN_Mps.py .

# Install system dependencies (for matplotlib)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install required Python packages
RUN pip install --no-cache-dir torch matplotlib

# Command to run your script
CMD ["python", "-u", "NN_Mps.py"]

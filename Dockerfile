# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install tesseract-ocr and libtesseract-dev
RUN apt-get update && \
    apt-get install -y tesseract-ocr libtesseract-dev iputils-ping && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Copy environment variables file
COPY .env /app/.env

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Script to find host IP and set it as an environment variable
RUN echo "#!/bin/sh\nHOST_IP=\$(/sbin/ip route|awk '/default/ { print \$3 }')\nexport HOST_IP\nexec \"\$@\"" > /app/start.sh && chmod +x /app/start.sh

# Run the FastAPI server with the script
CMD ["/app/start.sh", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

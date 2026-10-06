# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /workspace

# Install system dependencies required by OpenCV and Ultralytics
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY app/requirements.txt ./app/

# Install the Python dependencies
RUN pip install --no-cache-dir -r app/requirements.txt

# Copy the rest of the application code into the container
COPY app/ ./app/

# Set the working directory to where the execution script lives
WORKDIR /workspace/app

# Command to run the application natively
CMD ["python", "-u", "main.py"]
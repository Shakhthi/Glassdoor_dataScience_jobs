FROM python:3.11-slim-bookworm

LABEL maintainer = "MK"
LABEL version = "1.0"
LABEL description = "Python 3.11 Slim Bookworm Docker Image"

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY . /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["python", "app.py"]
# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files into the container
COPY app.py requirements.txt ./

# Fix 1: Create cache directory with permissions
RUN mkdir -p /app/cache && chmod -R 777 /app/cache

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Fix 2: Expose port 7860
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]

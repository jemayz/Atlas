# Use Python 3.11
FROM python:3.11-slim

WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# EXPOSE the port Render will use
EXPOSE 7860

# Use Gunicorn to run the app. It will get the port from the $PORT env var.
CMD ["gunicorn", "-b", "0.0.0.0:${PORT}", "app:app", "--timeout", "300"]
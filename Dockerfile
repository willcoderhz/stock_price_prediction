# Use the official Python image
FROM python:3.12-slim

# Set the working directory
WORKDIR /usr/src/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container
COPY . .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the default Django port
EXPOSE 8000

# Start the Gunicorn server for production
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "stock_price_prediction.wsgi:application"]

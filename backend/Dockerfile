FROM python:3.11-slim

# Install system dependencies for Chrome
RUN apt-get update && apt-get install -y \
    chromium chromium-driver \
    fonts-liberation libappindicator3-1 libasound2 libatk-bridge2.0-0 libatk1.0-0 \
    libcups2 libdbus-1-3 libgdk-pixbuf2.0-0 libnspr4 libnss3 libx11-xcb1 \
    libxcomposite1 libxdamage1 libxrandr2 xdg-utils libu2f-udev \
    libgbm1 libgtk-3-0 wget curl unzip gnupg \
    --no-install-recommends && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set environment for Chrome and Python
ENV CHROME_BIN=/usr/bin/chromium
ENV PATH=$PATH:/usr/lib/chromium:/usr/bin
ENV PYTHONPATH=/backend
ENV PYTHONUNBUFFERED=1

# Optional: Prevent Chrome from throwing X errors (good in headless/Docker)
ENV DISPLAY=:99

# Set working directory
WORKDIR /backend

COPY requirements.txt . 

RUN pip install -r requirements.txt 


# Copy app code
COPY . .

# Expose FastAPI port
EXPOSE 5000

# Run the app with hot reload (FastAPI dev mode)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000", "--reload"]

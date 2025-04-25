FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

# Copy the entire notebook directory
COPY . /app/notebook/

# Make sure we're in the right directory
WORKDIR /app/notebook

EXPOSE 7000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7000", "--reload"]
D
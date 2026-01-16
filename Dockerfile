FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Create all required directories including processed and logs
RUN mkdir -p data/raw data/processed reports/excel reports/pdf
CMD ["python", "main.py"]
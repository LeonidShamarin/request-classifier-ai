FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/
COPY main.py .

RUN mkdir -p output

# GEMINI_API_KEY передавати через `docker run -e GEMINI_API_KEY=...` або --env-file .env
ENTRYPOINT ["python", "main.py"]
CMD ["--input", "data/input_requests.csv", "--output-dir", "output"]

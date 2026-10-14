FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PORT=8000
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -m ml.pipeline > /dev/null
EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]

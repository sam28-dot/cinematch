FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PORT=8000
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -c "import urllib.request,zipfile,io;zipfile.ZipFile(io.BytesIO(urllib.request.urlopen('https://files.grouplens.org/datasets/movielens/ml-100k.zip',timeout=60).read())).extractall('data/raw')" || echo "MovieLens download failed - using synthetic sample"
RUN python -m ml.pipeline > /dev/null && python -m ml.evaluate > /dev/null
EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
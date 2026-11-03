FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py prompts.tsv ./
# Mount a Coolify persistent volume at /data: audio and the database live there.
ENV DATA_DIR=/data ALLOWED_ORIGIN=https://dani.sanchaya.net
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health')"
# One worker: SQLite is single-writer.
CMD ["gunicorn", "-w", "1", "--threads", "4", "-b", "0.0.0.0:8000", "app:app"]

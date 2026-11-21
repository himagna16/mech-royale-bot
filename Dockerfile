FROM python:3.13-slim
RUN pip install --no-cache-dir numpy==2.2.6
WORKDIR /bot
COPY . .
RUN python3 -m compileall -q /bot
ENV PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ENTRYPOINT ["python3", "-u", "/bot/main.py"]

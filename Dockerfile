# 日本リージョンのコンテナ host(Cloud Run 東京 / Fly.io nrt 等)向け。
# serving に不要な lightgbm/pytest は入れない(軽量・ビルド高速・libgomp不要)。
FROM python:3.12-slim
WORKDIR /app
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt
COPY keiba/ keiba/
COPY web/ web/
COPY model.pkl .
ENV PORT=8080
# enrich(各馬の過去走取得)が遅いので worker timeout を長めに。
CMD ["sh", "-c", "gunicorn web.app:app --workers 1 --threads 4 --timeout 180 --bind 0.0.0.0:$PORT"]

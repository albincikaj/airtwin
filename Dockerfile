FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev

COPY src ./src
COPY data ./data
COPY models ./models

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uv run streamlit run src/app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true"]

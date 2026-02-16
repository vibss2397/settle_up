FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directory structure
RUN mkdir -p src/llm src/creds

# Copy only what's needed for the app
COPY main.py .
COPY src/__init__.py src/
COPY src/mapping.json src/
COPY src/llm/__init__.py src/llm/
COPY src/llm/gemini_handler.py src/llm/
COPY src/llm/sheets_handler.py src/llm/
COPY src/llm/message_preprocessor.py src/llm
COPY src/creds/service_account.json src/creds/

ENV PORT=8080
ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["functions-framework", "--target=whatsapp_webhook", "--port=8080"]

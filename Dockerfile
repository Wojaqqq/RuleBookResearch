FROM python:3.10-slim

WORKDIR /app

COPY src/ /app/
COPY requirements.txt /app/

RUN pip install -r /app/requirements.txt

# Disable Python output buffering so logs appear immediately in the console
ENV PYTHONUNBUFFERED=1

CMD ["python3", "chat_request_handler.py"]

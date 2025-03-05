FROM python:3.10-slim

WORKDIR /app

COPY src/ /app/

RUN pip install -r /app/requirements.txt

ENV PYTHONUNBUFFERED=1

CMD ["python3", "chat_request_handler.py"]

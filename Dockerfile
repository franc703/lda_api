FROM python:3.10

RUN mkdir -p /app
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]



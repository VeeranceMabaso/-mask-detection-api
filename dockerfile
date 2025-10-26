FROM tensorflow/tensorflow:2.13.0
WORKDIR /app
COPY . .
RUN pip install fastapi uvicorn pillow python-multipart
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
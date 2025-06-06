FROM python:3.11.3
COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT [ "streamlit","run"]
CMD ["app.py"]
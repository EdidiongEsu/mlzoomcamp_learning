FROM python:3.8.12-slim

RUN pip install pipenv
RUN pip install waitress

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

##instsll pipenv and create a virtual environment

RUN pipenv install --system --deploy

COPY ["predict.py", "model_C=1.0.bin", "./"]

EXPOSE 9696

##breakdown waitress serve deployment into text chunks

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "predict:app"]
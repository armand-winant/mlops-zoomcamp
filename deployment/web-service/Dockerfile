FROM python:3.9-slim

# install dependencies
RUN pip install -U pip
RUN pip install pipenv

# set working directory
WORKDIR /app

# copy project files
COPY ["requirements.txt", "./"]

RUN pipenv install -r requirements.txt --system --deploy

COPY ["predict.py", "lin_reg.bin", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]
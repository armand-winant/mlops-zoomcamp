FROM python:3.9-slim

# Install dependencies
RUN pip install -U pip
RUN pip install pipenv

# Set working directory
WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

# Copy project files
COPY ["predict.py", "lin_reg.bin", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
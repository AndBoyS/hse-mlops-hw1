FROM python:3.9-slim-bullseye
ENV POETRY_VERSION=1.2.2
RUN pip install poetry==$POETRY_VERSION
COPY . /home
WORKDIR /home
RUN poetry config virtualenvs.create false
RUN poetry install --no-root
EXPOSE 5000
ENTRYPOINT FLASK_APP=main.py flask run --host=0.0.0.0
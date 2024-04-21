FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

COPY --chown=root:root . /root/server

WORKDIR /root

RUN apt update && \
    apt install -y python3-dev libmysqlclient-dev build-essential pkg-config && \
    apt clean && \
    pip install -r ./server/requirements.txt  --no-cache-dir
#    python manage.py makemigrations && \
#    python manage.py migrate && \


CMD ["python", "/root/server/manage.py", "runserver", "0.0.0.0:8000"]

EXPOSE 8000


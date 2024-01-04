FROM tensorflow/tensorflow:latest-gpu

COPY --chown=root:root . /root/

WORKDIR /root

RUN apt update && \
    apt install -y python3-dev default-libmysqlclient-dev build-essential pkg-config && \
    pip install -r requirements.txt && \
    mv isrgrootx1.pem /etc/ssl/certs

EXPOSE 8000


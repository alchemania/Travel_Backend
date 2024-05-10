FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

COPY --chown=root:root . /root/server

WORKDIR /root

RUN apt update && \
    apt install -y python3-dev libmysqlclient-dev build-essential pkg-config supervisor && \
    apt clean && \
    pip install -r ./server/requirements.txt --no-cache-dir

# Set an environment variable to indicate that this is running in a Docker container
ENV DEPLOYED=1

# Supervisord configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

EXPOSE 8000

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

# STAGE 1
FROM node:22 AS builder

# 设置工作目录
WORKDIR /app
# 克隆 GitHub 仓库
RUN git clone https://github.com/jiadol/Travel_Frontend.git .
# 安装依赖
RUN npm install
# 构建项目
RUN npm run build

# STAGE 2
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

COPY --chown=root:root . /root/server
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY --from=builder /app/dist /root/server/templates/dist

WORKDIR /root

# 设置环境变量
ENV AIRFLOW_HOME=/root/server
ENV AIRFLOW__CORE__DAGS_FOLDER=${AIRFLOW_HOME}/dags
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=sqlite:////${AIRFLOW_HOME}/airflow.db
ENV DJANGO_DEPLOYED=1

RUN apt update && \
    apt install -y python3-dev libmysqlclient-dev build-essential pkg-config supervisor && \
    apt clean && \
    pip install -r ./server/requirements.txt --no-cache-dir


EXPOSE 8000
EXPOSE 8080

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

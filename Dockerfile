# 使用Python 3.11官方镜像作为基础镜像
FROM hamgua/alpha-trading-bot-okx:base_alpine-v1.5.0

# 设置时区
ENV TZ=Asia/Shanghai

# 创建非root用户提高安全性
RUN mkdir -p /app/logs /app/data_json

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY  . .

# 暴露Streamlit端口
#EXPOSE 8501

# 健康检查简化
#HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
#    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# 使用exec形式确保信号传递
ENTRYPOINT ["python", "-u", "main.py"]

# docker 构建业务镜像命令
# docker buildx build --platform linux/amd64 --no-cache -t hamgua/alpha-trading-bot-okx:v3.0.8 ./
# docker push hamgua/alpha-trading-bot-okx:v3.0.8

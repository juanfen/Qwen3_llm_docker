#!/bin/bash

# 构建镜像
docker build -t mem-nostream-llm:latest .

# 运行容器（使用host网络模式，以便访问192.168.99.105:8194）
docker run -d \
  --name mem-nostream-llm \
  --network host \
  --restart unless-stopped \
  -e CUDA_VISIBLE_DEVICES=1 \
  -v /mnt/Data/jungao/Qwen3:/mnt/Data/jungao/Qwen3:ro \
  mem-nostream-llm:latest

# 查看日志
docker logs -f mem-nostream-llm

# 停止容器
# docker stop mem-nostream-llm

# 启动容器
# docker start mem-nostream-llm

# 删除容器
# docker rm mem-nostream-llm
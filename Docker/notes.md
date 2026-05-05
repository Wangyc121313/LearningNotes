# Docker

> 容器化与镜像管理学习笔记

## 内容概览

- 核心概念（Image、Container、Volume、Network）
- Dockerfile 编写
- Docker Compose 多容器编排
- 常用命令速查
- GPU 容器（nvidia-docker）

## 1. 核心概念

| 概念 | 说明 |
|------|------|
| **Image（镜像）** | 只读模板，包含完整的运行环境（OS + 依赖 + 代码） |
| **Container（容器）** | 镜像的运行实例，相互隔离，启停极快 |
| **Volume（卷）** | 持久化数据存储，容器删除后数据不丢失 |
| **Network** | 容器间通信，默认 bridge 网络 |
| **Registry** | 镜像仓库（Docker Hub、阿里云、私有 Harbor） |

**Docker vs VM**：Docker 共享宿主机内核，无需完整 OS，启动秒级；VM 有完整 OS，启动分钟级，隔离更强。

---

## 2. 常用命令速查

### 镜像操作
```bash
docker pull ubuntu:22.04              # 拉取镜像
docker images                          # 列出本地镜像
docker rmi ubuntu:22.04               # 删除镜像
docker build -t myapp:v1 .            # 从 Dockerfile 构建
docker tag myapp:v1 registry/myapp:v1 # 打标签
docker push registry/myapp:v1         # 推送到仓库
```

### 容器操作
```bash
docker run -it ubuntu:22.04 bash      # 交互式运行
docker run -d --name web nginx        # 后台运行，命名
docker run -p 8080:80 nginx           # 端口映射 宿主:容器
docker run -v /host/data:/data nginx  # 挂载卷
docker ps                              # 列出运行中容器
docker ps -a                           # 列出所有容器
docker stop web && docker rm web      # 停止并删除
docker exec -it web bash              # 进入运行中容器
docker logs -f web                    # 跟踪日志
docker cp file.txt web:/app/          # 拷贝文件进容器
docker inspect web                    # 查看详细信息
```

### 系统清理
```bash
docker system prune -af               # 清理所有未使用资源
docker volume prune                    # 清理未用卷
docker image prune                     # 清理悬空镜像
```

---

## 3. Dockerfile 编写

```dockerfile
# 1. 选择基础镜像（越小越好：优先 slim/alpine）
FROM python:3.11-slim

# 2. 设置工作目录
WORKDIR /app

# 3. 先复制依赖文件（利用层缓存，依赖不变则不重建）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 再复制代码
COPY . .

# 5. 暴露端口（文档用途）
EXPOSE 8000

# 6. 非 root 用户运行（安全最佳实践）
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# 7. 启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Dockerfile 最佳实践**：
- 合并 RUN 命令（减少层数）：`RUN apt-get update && apt-get install -y pkg && rm -rf /var/lib/apt/lists/*`
- 使用 `.dockerignore` 排除 `.git`、`__pycache__`、`*.pyc`
- 多阶段构建（Multi-stage Build）减少最终镜像大小

### 多阶段构建示例
```dockerfile
# 构建阶段
FROM python:3.11 AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

# 运行阶段（只复制必要文件）
FROM python:3.11-slim
COPY --from=builder /install /usr/local
COPY app/ /app
CMD ["python", "/app/main.py"]
```

---

## 4. Docker Compose（多容器编排）

```yaml
# docker-compose.yml
version: "3.9"

services:
  web:
    build: .                          # 从当前目录 Dockerfile 构建
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/mydb
    volumes:
      - ./app:/app                    # 开发时热重载
    depends_on:
      db:
        condition: service_healthy    # 等待 db 健康检查通过
    restart: unless-stopped

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: mydb
    volumes:
      - postgres_data:/var/lib/postgresql/data   # 持久化
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "user"]
      interval: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

```bash
docker compose up -d            # 后台启动所有服务
docker compose down -v          # 停止并删除（-v 同时删卷）
docker compose logs -f web      # 跟踪 web 服务日志
docker compose exec web bash    # 进入 web 容器
docker compose ps               # 查看服务状态
```

---

## 5. GPU 容器（nvidia-docker）

```bash
# 前提：安装 NVIDIA Container Toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# 运行时挂载所有 GPU
docker run --gpus all nvidia/cuda:12.2-base-ubuntu22.04 nvidia-smi

# 只使用 GPU 0,1
docker run --gpus '"device=0,1"' myapp

# docker-compose 中
services:
  trainer:
    image: pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

**常用 AI 基础镜像**：
- `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime` — PyTorch 官方
- `nvcr.io/nvidia/tensorrt:24.01-py3` — TensorRT
- `nvcr.io/nvidia/tritonserver:24.01-py3` — Triton 推理服务器

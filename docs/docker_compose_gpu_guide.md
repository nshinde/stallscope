# Docker Compose + NVIDIA Container Toolkit Guide

This setup runs stallscope periodically in a container with GPU access.

## 1) Install NVIDIA Container Toolkit (host)

Follow NVIDIA's official installation guide for your OS:

- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

After installation, verify Docker can see GPUs:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## 2) Build + run with Docker Compose

```bash
docker compose up --build -d
```

This uses:

- `gpus: all` for GPU access
- `/proc:/host_proc:ro` so network/system metrics read from host procfs
- periodic mode (`--interval-seconds 15`)
- Prometheus textfile output at `./artifacts/stallscope.prom`

## 3) View logs

```bash
docker compose logs -f stallscope
```

## 4) Customize alert hooks

Example with webhook alerts:

```yaml
services:
  stallscope:
    command:
      - --interval-seconds
      - "15"
      - --alert-webhook-url
      - http://alertmanager:9093/api/v2/alerts
```

## 5) Stop

```bash
docker compose down
```

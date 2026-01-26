FROM python:3.11-slim

ENV LANG=C.UTF-8 \
    PYTHONUNBUFFERED=1

# Copy filesystem
COPY rootfs/ /

WORKDIR /opt/app

# Install Python dependencies
RUN pip install --no-cache-dir -r /opt/app/requirements.txt

EXPOSE 8000

# Ensure service script is executable
RUN chmod +x /etc/services.d/server/run

# Default command for local runs (HA builder overrides entrypoint/init)
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

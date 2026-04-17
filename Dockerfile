# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile
# Inventory Optimization Platform — RHEL 10 (UBI 10) base
# ─────────────────────────────────────────────────────────────────────────────

# Red Hat Universal Base Image 10 (Python 3.11 minimal)
FROM registry.access.redhat.com/ubi10/python-311:latest

# ── Metadata ─────────────────────────────────────────────────────────────────
LABEL maintainer="Data Science Team" \
      version="1.0.0" \
      description="Inventory Optimisation via Time-Series Demand Forecasting" \
      org.opencontainers.image.title="inventory-intelligence" \
      org.opencontainers.image.base.name="registry.access.redhat.com/ubi10/python-311"

# ── Security: run as non-root user (UID 1001 is default in UBI) ──────────────
USER 0
RUN dnf install -y gcc gcc-c++ make libgomp && \
    dnf clean all && \
    rm -rf /var/cache/dnf

USER 1001

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (layer-cached) ───────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY engine.py .
COPY app.py    .

# ── Streamlit configuration ───────────────────────────────────────────────────
RUN mkdir -p /app/.streamlit
COPY streamlit_config.toml /app/.streamlit/config.toml

# ── Expose Streamlit port ─────────────────────────────────────────────────────
EXPOSE 8501

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]

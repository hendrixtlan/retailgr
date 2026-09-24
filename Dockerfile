# The serving image. One stage to build the wheel, one to run it, so the
# runtime carries no compiler and no build cache.
#
# Nothing here names a cloud. The image runs the same on any registry and any
# cluster; where the bundle and the online store live is configuration, which
# is the property `retailgr audit` checks and `deploy/k8s/` supplies.

FROM python:3.11-slim AS build
WORKDIR /src
RUN pip install --no-cache-dir build
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m build --wheel --outdir /dist

FROM python:3.11-slim AS runtime

# A numeric, unprivileged user: the manifests set runAsNonRoot, which the
# kubelet can only enforce when the image does not start as root.
RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin retailgr

WORKDIR /app
COPY --from=build /dist/*.whl /tmp/
# --no-compile keeps the layer smaller; the .pyc cost is paid once at the
# cold start this project measures at ~1.1 s, which is inside the readiness
# budget in deploy/k8s/deployment.yaml.
RUN pip install --no-cache-dir --no-compile /tmp/*.whl "uvicorn[standard]" \
    && rm -rf /tmp/*.whl

COPY configs ./configs

# Threads are pinned rather than left to OpenMP's default, which reads the
# *node's* core count and ignores the pod's CPU request — the reliable way to
# get 64 BLAS threads fighting over one core.
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RETAILGR_BUNDLE=/bundle

USER 10001
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8080/healthz')"

ENTRYPOINT ["retailgr", "serve"]
CMD ["--host", "0.0.0.0", "--port", "8080"]

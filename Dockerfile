# The one-command demo in a container: builds a small point-in-time store on free data,
# runs a light (no-LLM) pass and serves the dashboard on port 8502.
#
#   docker build -t claude-trader .
#   docker run --rm -it -p 8502:8502 -e SEC_CONTACT_EMAIL=you@example.com \
#       -v claude-trader-demo:/data claude-trader
#
# No secrets are baked in (.dockerignore keeps .env, databases and data out of the
# build context); pass credentials at run time with -e. See docs/SETUP.md "Docker".
FROM python:3.12-slim

# libgomp1: OpenMP runtime the lightgbm wheel links against (absent from slim images).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Non-root. The code stays root-owned (read-only to the app); all state goes to /data,
# which is created owned by the app user so a named volume mounted there inherits it.
RUN useradd --create-home --uid 10001 trader \
    && mkdir -p /data \
    && chown trader:trader /data
USER trader

ENV CLAUDE_TRADER_HOME=/data
VOLUME ["/data"]
EXPOSE 8502

# demo.py turns Ctrl+C (SIGINT) into a clean shutdown of the dashboard process; make
# `docker stop` send the same signal instead of SIGTERM, which PID 1 would ignore.
STOPSIGNAL SIGINT

ENTRYPOINT ["python", "-m", "scripts.demo", "--home", "/data", "--host", "0.0.0.0", "--no-browser"]
CMD []

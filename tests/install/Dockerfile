# Install test for the npm/npx/pnpm launcher and the light Python install.
# Build from the repository root after `uv build`:
#   docker build -f tests/install/Dockerfile -t ragmcp-install-test .
#   docker run --rm ragmcp-install-test
FROM node:20-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl python3 \
    && rm -rf /var/lib/apt/lists/*

# uv lives outside the default PATH so one test can hide it on purpose.
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/opt/uv/bin sh
ENV PATH="/opt/uv/bin:${PATH}"

RUN corepack enable && corepack prepare pnpm@latest --activate

WORKDIR /work
COPY dist/ragmcp-*.whl /work/
COPY npm/ /work/npm/
COPY tests/install/ /work/install/
RUN cd /work/npm && npm pack --pack-destination /work >/dev/null && ls /work

ENTRYPOINT ["bash", "/work/install/smoke.sh"]

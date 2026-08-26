# ── Build stage ──────────────────────────────────────────────
FROM rust:1.80-slim-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/

RUN cargo build --release --bin hypno-cli --bin hypno-convert \
    && strip target/release/hypno-cli target/release/hypno-convert

# ── Runtime stage ─────────────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/hypno-cli /usr/local/bin/
COPY --from=builder /build/target/release/hypno-convert /usr/local/bin/

RUN mkdir -p /models /data

ENV HYPNO_MODELS_DIR=/models
ENV HYPNO_DATA_DIR=/data

VOLUME ["/models", "/data"]

ENTRYPOINT ["hypno-cli"]

# -- Build stage --------------------------------------------------------------
FROM rust:1-slim-bookworm AS builder

WORKDIR /build

# Cache dependency compilation
COPY rootfs/opt/app/Cargo.toml ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs \
    && cargo build --release \
    && rm -rf src target/release/.fingerprint/tflite-server-*

# Build actual source
COPY rootfs/opt/app/src/ src/
RUN cargo build --release

# -- Runtime stage ------------------------------------------------------------
FROM debian:bookworm-slim

COPY --from=builder /build/target/release/tflite-server /usr/local/bin/tflite-server

# Copy service script and static assets (no Rust source needed at runtime)
COPY rootfs/etc/ /etc/
COPY rootfs/opt/app/static/ /opt/app/static/

RUN chmod +x /etc/services.d/server/run

EXPOSE 8000

# Default command for local runs (HA builder overrides entrypoint/init)
CMD ["/usr/local/bin/tflite-server"]

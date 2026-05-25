FROM debian:bookworm AS build

ARG RINHA_LEAF_SIZE=24
ARG RINHA_EARLY_DISTANCE_MILLI=160
ARG CFLAGS_EXTRA="-O3 -march=haswell -mtune=haswell -mavx2 -flto -fno-plt -DNDEBUG -std=c11 -Wall -Wextra -Wshadow"

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gzip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src/c-api-rinha2026
COPY . .
RUN make clean all CFLAGS="${CFLAGS_EXTRA} -DRINHA_EARLY_DISTANCE_MILLI=${RINHA_EARLY_DISTANCE_MILLI}"
RUN ./build-index resources/references.json.gz data/knn.idx "${RINHA_LEAF_SIZE}"
RUN strip -s api lb

FROM debian:bookworm-slim AS runtime
WORKDIR /app
COPY --from=build /src/c-api-rinha2026/api /app/api
COPY --from=build /src/c-api-rinha2026/lb /app/lb
COPY --from=build /src/c-api-rinha2026/data/knn.idx /app/data/knn.idx

CMD ["/app/api"]

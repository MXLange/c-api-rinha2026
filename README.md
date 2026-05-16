# c-api-rinha2026

Versao C da API/LB com o mesmo algoritmo kNN exato usado no projeto Go.

Principais escolhas:

- build-time gera `data/knn.idx` a partir de `resources/references.json.gz`;
- runtime usa `mmap` no indice e copia apenas metadados pequenos de particoes/nos;
- LB passa o FD TCP para a API via Unix socket usando `SCM_RIGHTS`;
- API opera no FD recebido com `read`/`write`, parser posicional e respostas HTTP fixas;
- compilacao padrao usa `-O3 -flto -march=haswell -mavx2`, mirando somente o Mac Mini Late 2014/Haswell com AVX2.
- early return preservando K=5 quando os 5 melhores vizinhos ja estao abaixo de distancia normalizada `0.14`.

Build local:

```sh
make clean all
./build-index resources/references.json.gz data/knn.idx 128
```

Build da imagem:

```sh
docker build -t mxlange/c-api-rinha2026:latest \
  --build-arg RINHA_LEAF_SIZE=128 \
  --build-arg RINHA_EARLY_DISTANCE_MILLI=140 \
  .
```

Execucao:

```sh
docker compose up
```

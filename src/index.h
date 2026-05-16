#ifndef RINHA_INDEX_H
#define RINHA_INDEX_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "common.h"

typedef struct {
    uint32_t key;
    int32_t root;
    int32_t length;
    QueryVector min;
    QueryVector max;
} Partition;

typedef struct {
    int32_t left;
    int32_t right;
    int32_t start;
    int32_t len;
    QueryVector min;
    QueryVector max;
} Node;

typedef struct {
    uint8_t *data;
    size_t size;
    Partition *partitions;
    Node *nodes;
    const int16_t *vectors;
    const uint8_t *labels;
    int part_count;
    int node_count;
    int block_count;
    int part_by_key[256];
} Index;

bool index_open(Index *idx, const char *path);
void index_close(Index *idx);
uint8_t index_predict_fraud_count(const Index *idx, const QueryVector query);

#endif

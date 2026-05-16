#ifndef RINHA_COMMON_H
#define RINHA_COMMON_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define RINHA_DIMS 14
#define RINHA_PACKED_DIMS 16
#define RINHA_SCALE 10000
#define RINHA_K 5
#define RINHA_LANES 8

#define RINHA_MAGIC "GOKNN001"
#define RINHA_HEADER_SIZE 64
#define RINHA_PART_SIZE 76
#define RINHA_NODE_SIZE 80

typedef int16_t QueryVector[RINHA_PACKED_DIMS];

static inline uint32_t read_u32le(const uint8_t *p) {
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static inline int32_t read_i32le(const uint8_t *p) {
    return (int32_t)read_u32le(p);
}

static inline int16_t read_i16le(const uint8_t *p) {
    return (int16_t)(((uint16_t)p[0]) | ((uint16_t)p[1] << 8));
}

static inline void write_u16le(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
}

static inline void write_u32le(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

static inline uint32_t partition_key(const QueryVector v) {
    uint32_t key = 0;
    if (v[5] >= 0) key |= 1u << 0;
    if (v[9] > 0) key |= 1u << 1;
    if (v[10] > 0) key |= 1u << 2;
    if (v[11] > 0) key |= 1u << 3;

    if (v[12] <= 2047) {
    } else if (v[12] <= 4095) {
        key |= 1u << 4;
    } else if (v[12] <= 6143) {
        key |= 2u << 4;
    } else {
        key |= 3u << 4;
    }

    if (v[2] > 4096) key |= 1u << 6;
    if (v[8] > 2048) key |= 1u << 7;
    return key;
}

static inline int64_t lower_bound_dim(int16_t q, int16_t min, int16_t max) {
    int64_t diff = 0;
    if (q < min) {
        diff = (int64_t)min - (int64_t)q;
    } else if (q > max) {
        diff = (int64_t)q - (int64_t)max;
    }
    return diff * diff;
}

static inline int64_t lower_bound_vec(const QueryVector q, const QueryVector min, const QueryVector max) {
    return lower_bound_dim(q[0], min[0], max[0]) +
           lower_bound_dim(q[1], min[1], max[1]) +
           lower_bound_dim(q[2], min[2], max[2]) +
           lower_bound_dim(q[3], min[3], max[3]) +
           lower_bound_dim(q[4], min[4], max[4]) +
           lower_bound_dim(q[5], min[5], max[5]) +
           lower_bound_dim(q[6], min[6], max[6]) +
           lower_bound_dim(q[7], min[7], max[7]) +
           lower_bound_dim(q[8], min[8], max[8]) +
           lower_bound_dim(q[9], min[9], max[9]) +
           lower_bound_dim(q[10], min[10], max[10]) +
           lower_bound_dim(q[11], min[11], max[11]) +
           lower_bound_dim(q[12], min[12], max[12]) +
           lower_bound_dim(q[13], min[13], max[13]);
}

static inline void insert_best(int64_t dist, uint8_t label, int64_t best_dists[RINHA_K], uint8_t best_labels[RINHA_K]) {
    if (dist >= best_dists[RINHA_K - 1]) {
        return;
    }
    int pos = RINHA_K - 1;
    while (pos > 0 && dist < best_dists[pos - 1]) {
        best_dists[pos] = best_dists[pos - 1];
        best_labels[pos] = best_labels[pos - 1];
        pos--;
    }
    best_dists[pos] = dist;
    best_labels[pos] = label;
}

#endif

#define _GNU_SOURCE

#include "index.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if !defined(__AVX2__)
#error "c-api-rinha2026 requires AVX2. Build with -march=haswell or -mavx2."
#endif

#include <immintrin.h>

#ifndef RINHA_EARLY_DISTANCE_MILLI
#define RINHA_EARLY_DISTANCE_MILLI 140
#endif

typedef struct {
    int idx;
    int64_t bound;
} Candidate;

static const int64_t early_distance_limit =
    ((int64_t)RINHA_SCALE * RINHA_EARLY_DISTANCE_MILLI / 1000) *
    ((int64_t)RINHA_SCALE * RINHA_EARLY_DISTANCE_MILLI / 1000);

static void read_vector(const uint8_t *buf, QueryVector out) {
    for (int i = 0; i < RINHA_PACKED_DIMS; i++) {
        out[i] = read_i16le(buf + i * 2);
    }
}

static void read_partition(const uint8_t *buf, Partition *out) {
    out->key = read_u32le(buf);
    out->root = read_i32le(buf + 4);
    out->length = read_i32le(buf + 8);
    read_vector(buf + 12, out->min);
    read_vector(buf + 44, out->max);
}

static void read_node(const uint8_t *buf, Node *out) {
    out->left = read_i32le(buf);
    out->right = read_i32le(buf + 4);
    out->start = read_i32le(buf + 8);
    out->len = read_i32le(buf + 12);
    read_vector(buf + 16, out->min);
    read_vector(buf + 48, out->max);
}

bool index_open(Index *idx, const char *path) {
    memset(idx, 0, sizeof(*idx));
    int fd = open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return false;

    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < RINHA_HEADER_SIZE) {
        close(fd);
        return false;
    }

    size_t size = (size_t)st.st_size;
    uint8_t *data = mmap(NULL, size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (data == MAP_FAILED) return false;

    if (memcmp(data, RINHA_MAGIC, 8) != 0 ||
        read_u32le(data + 8) != RINHA_SCALE ||
        read_u32le(data + 12) != RINHA_DIMS ||
        read_u32le(data + 16) != RINHA_PACKED_DIMS ||
        read_u32le(data + 20) != RINHA_LANES) {
        munmap(data, size);
        return false;
    }

    int part_count = (int)read_u32le(data + 28);
    int node_count = (int)read_u32le(data + 32);
    int block_count = (int)read_u32le(data + 36);
    if (part_count < 0 || node_count < 0 || block_count < 0) {
        munmap(data, size);
        return false;
    }

    size_t parts_off = RINHA_HEADER_SIZE;
    size_t nodes_off = parts_off + (size_t)part_count * RINHA_PART_SIZE;
    size_t vectors_off = nodes_off + (size_t)node_count * RINHA_NODE_SIZE;
    size_t vector_bytes = (size_t)block_count * RINHA_DIMS * RINHA_LANES * sizeof(int16_t);
    size_t labels_off = vectors_off + vector_bytes;
    size_t labels_end = labels_off + (size_t)block_count * RINHA_LANES;
    if (labels_end > size) {
        munmap(data, size);
        return false;
    }

    Partition *parts = malloc((size_t)part_count * sizeof(Partition));
    Node *nodes = malloc((size_t)node_count * sizeof(Node));
    if ((part_count > 0 && parts == NULL) || (node_count > 0 && nodes == NULL)) {
        free(parts);
        free(nodes);
        munmap(data, size);
        return false;
    }

    for (int i = 0; i < part_count; i++) {
        read_partition(data + parts_off + (size_t)i * RINHA_PART_SIZE, &parts[i]);
    }
    for (int i = 0; i < node_count; i++) {
        read_node(data + nodes_off + (size_t)i * RINHA_NODE_SIZE, &nodes[i]);
    }

#ifdef MADV_HUGEPAGE
    (void)madvise(data + vectors_off, labels_end - vectors_off, MADV_HUGEPAGE);
#endif
    (void)madvise(data + vectors_off, labels_end - vectors_off, MADV_WILLNEED);

    idx->data = data;
    idx->size = size;
    idx->partitions = parts;
    idx->nodes = nodes;
    idx->vectors = (const int16_t *)(const void *)(data + vectors_off);
    idx->labels = data + labels_off;
    idx->part_count = part_count;
    idx->node_count = node_count;
    idx->block_count = block_count;
    for (int i = 0; i < 256; i++) {
        idx->part_by_key[i] = -1;
    }
    for (int i = 0; i < part_count; i++) {
        if (parts[i].key < 256) {
            idx->part_by_key[parts[i].key] = i;
        }
    }
    return true;
}

void index_close(Index *idx) {
    if (idx->data != NULL) {
        munmap(idx->data, idx->size);
    }
    free(idx->partitions);
    free(idx->nodes);
    memset(idx, 0, sizeof(*idx));
}

static inline void sort_candidates(Candidate *items, int count) {
    for (int i = 1; i < count; i++) {
        Candidate item = items[i];
        int j = i - 1;
        while (j >= 0 && items[j].bound > item.bound) {
            items[j + 1] = items[j];
            j--;
        }
        items[j + 1] = item;
    }
}

static inline void distance_block8(const int16_t *block, const QueryVector query, int64_t out[RINHA_LANES]) {
    __m256i acc_lo = _mm256_setzero_si256();
    __m256i acc_hi = _mm256_setzero_si256();
    for (int d = 0; d < RINHA_DIMS; d++) {
        const __m128i packed = _mm_loadu_si128((const __m128i *)(const void *)(block + d * RINHA_LANES));
        const __m256i values = _mm256_cvtepi16_epi32(packed);
        const __m256i q = _mm256_set1_epi32((int)query[d]);
        const __m256i diff = _mm256_sub_epi32(values, q);
        const __m256i sq = _mm256_mullo_epi32(diff, diff);
        const __m128i sq_lo = _mm256_castsi256_si128(sq);
        const __m128i sq_hi = _mm256_extracti128_si256(sq, 1);
        acc_lo = _mm256_add_epi64(acc_lo, _mm256_cvtepi32_epi64(sq_lo));
        acc_hi = _mm256_add_epi64(acc_hi, _mm256_cvtepi32_epi64(sq_hi));
    }
    _mm256_storeu_si256((__m256i *)(void *)out, acc_lo);
    _mm256_storeu_si256((__m256i *)(void *)(out + 4), acc_hi);
}

static inline bool early_done(const int64_t best_dists[RINHA_K]) {
    return best_dists[RINHA_K - 1] <= early_distance_limit;
}

static inline bool scan_leaf(const Index *idx, const Node *n, const QueryVector query,
                             int64_t best_dists[RINHA_K], uint8_t best_labels[RINHA_K]) {
    const int start_block = n->start;
    const int length = n->len;
    const int blocks = (length + RINHA_LANES - 1) / RINHA_LANES;

    for (int b = 0; b < blocks; b++) {
        const int block = start_block + b;
        const int base = block * RINHA_DIMS * RINHA_LANES;
        const int labels_base = block * RINHA_LANES;
        int lane_count = length - b * RINHA_LANES;
        if (lane_count > RINHA_LANES) lane_count = RINHA_LANES;

        int64_t dists[RINHA_LANES];
        distance_block8(idx->vectors + base, query, dists);
        for (int lane = 0; lane < lane_count; lane++) {
            insert_best(dists[lane], idx->labels[labels_base + lane], best_dists, best_labels);
        }
        if (early_done(best_dists)) return true;
    }
    return false;
}

static bool search_node(const Index *idx, int root, int64_t root_bound, const QueryVector query,
                        int64_t best_dists[RINHA_K], uint8_t best_labels[RINHA_K]) {
    if (root < 0 || root >= idx->node_count) return false;

    int stack_node[128];
    int64_t stack_bound[128];
    int stack_len = 0;
    int current = root;
    int64_t current_bound = root_bound;

    for (;;) {
        if (current_bound < best_dists[RINHA_K - 1]) {
            const Node *n = &idx->nodes[current];
            if (n->left < 0) {
                if (scan_leaf(idx, n, query, best_dists, best_labels)) return true;
            } else {
                const int left = n->left;
                const int right = n->right;
                const int64_t lb = lower_bound_vec(query, idx->nodes[left].min, idx->nodes[left].max);
                const int64_t rb = lower_bound_vec(query, idx->nodes[right].min, idx->nodes[right].max);

                if (lb <= rb) {
                    if (rb < best_dists[RINHA_K - 1] && stack_len < 128) {
                        stack_node[stack_len] = right;
                        stack_bound[stack_len] = rb;
                        stack_len++;
                    }
                    current = left;
                    current_bound = lb;
                    continue;
                }

                if (lb < best_dists[RINHA_K - 1] && stack_len < 128) {
                    stack_node[stack_len] = left;
                    stack_bound[stack_len] = lb;
                    stack_len++;
                }
                current = right;
                current_bound = rb;
                continue;
            }
        }

        if (stack_len == 0) break;
        stack_len--;
        current = stack_node[stack_len];
        current_bound = stack_bound[stack_len];
    }
    return early_done(best_dists);
}

uint8_t index_predict_fraud_count(const Index *idx, const QueryVector query) {
    int64_t best_dists[RINHA_K];
    uint8_t best_labels[RINHA_K] = {0, 0, 0, 0, 0};
    for (int i = 0; i < RINHA_K; i++) {
        best_dists[i] = INT64_MAX;
    }

    const uint32_t key = partition_key(query);
    const int match = idx->part_by_key[key & 255u];

    if (match >= 0) {
        if (search_node(idx, idx->partitions[match].root, 0, query, best_dists, best_labels)) {
            goto done;
        }
    }

    Candidate candidates[256];
    int count = 0;
    for (int i = 0; i < idx->part_count; i++) {
        if (i == match) continue;
        const int64_t bound = lower_bound_vec(query, idx->partitions[i].min, idx->partitions[i].max);
        if (bound >= best_dists[RINHA_K - 1]) continue;
        candidates[count].idx = i;
        candidates[count].bound = bound;
        count++;
    }
    sort_candidates(candidates, count);

    for (int i = 0; i < count; i++) {
        if (candidates[i].bound >= best_dists[RINHA_K - 1]) break;
        const Partition *part = &idx->partitions[candidates[i].idx];
        if (search_node(idx, part->root, candidates[i].bound, query, best_dists, best_labels)) {
            break;
        }
    }

done:
    uint8_t fraud = 0;
    for (int i = 0; i < RINHA_K; i++) {
        fraud += best_labels[i];
    }
    return fraud;
}

#define _GNU_SOURCE

#include "common.h"
#include "payload.h"

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

typedef struct {
    QueryVector vector;
    uint8_t label;
} Reference;

typedef struct {
    Reference *items;
    int len;
    int cap;
} RefVec;

typedef struct {
    int *items;
    int len;
    int cap;
} IntVec;

typedef struct {
    int32_t left;
    int32_t right;
    int start;
    int len;
    QueryVector min;
    QueryVector max;
} BuildNode;

typedef struct {
    BuildNode *items;
    int len;
    int cap;
} NodeVec;

typedef struct {
    uint32_t key;
    int root;
} PartRoot;

static const Reference *sort_refs;
static int sort_dim;

static void die(const char *message) {
    perror(message);
    exit(1);
}

static void *xmalloc(size_t size) {
    void *ptr = malloc(size);
    if (ptr == NULL) die("malloc");
    return ptr;
}

static void *xrealloc(void *ptr, size_t size) {
    void *next = realloc(ptr, size);
    if (next == NULL) die("realloc");
    return next;
}

static void ref_append(RefVec *v, Reference ref) {
    if (v->len == v->cap) {
        v->cap = v->cap == 0 ? 3100000 : v->cap * 2;
        v->items = xrealloc(v->items, (size_t)v->cap * sizeof(Reference));
    }
    v->items[v->len++] = ref;
}

static void int_append(IntVec *v, int item) {
    if (v->len == v->cap) {
        v->cap = v->cap == 0 ? 1024 : v->cap * 2;
        v->items = xrealloc(v->items, (size_t)v->cap * sizeof(int));
    }
    v->items[v->len++] = item;
}

static int node_append(NodeVec *v, BuildNode node) {
    if (v->len == v->cap) {
        v->cap = v->cap == 0 ? 131072 : v->cap * 2;
        v->items = xrealloc(v->items, (size_t)v->cap * sizeof(BuildNode));
    }
    v->items[v->len] = node;
    return v->len++;
}

static const uint8_t *find_bytes(const uint8_t *hay, const uint8_t *end, const char *needle, size_t needle_len) {
    if ((size_t)(end - hay) < needle_len) return NULL;
    const uint8_t first = (uint8_t)needle[0];
    const uint8_t *last = end - needle_len;
    for (const uint8_t *p = hay; p <= last; p++) {
        if (*p == first && memcmp(p, needle, needle_len) == 0) {
            return p;
        }
    }
    return NULL;
}

static bool read_gzip(const char *path, uint8_t **out, size_t *out_len) {
    int pipe_fd[2];
    if (pipe(pipe_fd) != 0) return false;

    pid_t pid = fork();
    if (pid < 0) {
        close(pipe_fd[0]);
        close(pipe_fd[1]);
        return false;
    }

    if (pid == 0) {
        close(pipe_fd[0]);
        if (dup2(pipe_fd[1], STDOUT_FILENO) < 0) _exit(127);
        close(pipe_fd[1]);
        execlp("gzip", "gzip", "-dc", path, (char *)NULL);
        _exit(127);
    }

    close(pipe_fd[1]);

    size_t cap = 64u * 1024u * 1024u;
    size_t len = 0;
    uint8_t *buf = xmalloc(cap + 1);
    for (;;) {
        if (cap - len < 1024u * 1024u) {
            cap *= 2;
            buf = xrealloc(buf, cap + 1);
        }
        ssize_t n = read(pipe_fd[0], buf + len, cap - len);
        if (n < 0) {
            if (errno == EINTR) continue;
            close(pipe_fd[0]);
            free(buf);
            waitpid(pid, NULL, 0);
            return false;
        }
        if (n == 0) break;
        len += (size_t)n;
    }
    close(pipe_fd[0]);
    int status = 0;
    if (waitpid(pid, &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        free(buf);
        return false;
    }
    buf[len] = 0;
    *out = buf;
    *out_len = len;
    return true;
}

static void parse_references(const char *path, RefVec *refs) {
    uint8_t *data = NULL;
    size_t len = 0;
    if (!read_gzip(path, &data, &len)) {
        die("read gzip");
    }

    const uint8_t *p = data;
    const uint8_t *end = data + len;
    const char vector_needle[] = "\"vector\":[";
    const char label_needle[] = "\"label\":\"";

    for (;;) {
        p = find_bytes(p, end, vector_needle, sizeof(vector_needle) - 1);
        if (p == NULL) break;
        p += sizeof(vector_needle) - 1;

        Reference ref;
        memset(&ref, 0, sizeof(ref));
        for (int d = 0; d < RINHA_DIMS; d++) {
            while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ',' || *p == '[')) {
                p++;
            }
            char *next = NULL;
            double value = strtod((const char *)p, &next);
            if (next == (const char *)p) {
                fprintf(stderr, "invalid vector value near offset %zu\n", (size_t)(p - data));
                exit(1);
            }
            ref.vector[d] = quantize_value(value);
            p = (const uint8_t *)next;
        }

        const uint8_t *label = find_bytes(p, end, label_needle, sizeof(label_needle) - 1);
        if (label == NULL) {
            fprintf(stderr, "missing label after offset %zu\n", (size_t)(p - data));
            exit(1);
        }
        label += sizeof(label_needle) - 1;
        ref.label = (uint8_t)(end - label >= 5 && memcmp(label, "fraud", 5) == 0);
        ref_append(refs, ref);
        p = label;
    }

    free(data);
}

static void bounds(const Reference *refs, const int *indices, int len, QueryVector min, QueryVector max) {
    for (int d = 0; d < RINHA_PACKED_DIMS; d++) {
        min[d] = INT16_MAX;
        max[d] = INT16_MIN;
    }
    for (int i = 0; i < len; i++) {
        const QueryVector *v = &refs[indices[i]].vector;
        for (int d = 0; d < RINHA_PACKED_DIMS; d++) {
            if ((*v)[d] < min[d]) min[d] = (*v)[d];
            if ((*v)[d] > max[d]) max[d] = (*v)[d];
        }
    }
}

static int widest_dimension(const QueryVector min, const QueryVector max) {
    int best_dim = 0;
    int32_t best_width = INT32_MIN;
    for (int d = 0; d < RINHA_DIMS; d++) {
        int32_t width = (int32_t)max[d] - (int32_t)min[d];
        if (width > best_width) {
            best_width = width;
            best_dim = d;
        }
    }
    return best_dim;
}

static int cmp_index_by_dim(const void *a, const void *b) {
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    int16_t va = sort_refs[ia].vector[sort_dim];
    int16_t vb = sort_refs[ib].vector[sort_dim];
    return (va > vb) - (va < vb);
}

static int build_tree(const Reference *refs, const int *indices, int len, int leaf_size, RefVec *blocks, NodeVec *nodes) {
    QueryVector min, max;
    bounds(refs, indices, len, min, max);

    BuildNode placeholder;
    memset(&placeholder, 0, sizeof(placeholder));
    placeholder.left = -1;
    placeholder.right = -1;
    memcpy(placeholder.min, min, sizeof(QueryVector));
    memcpy(placeholder.max, max, sizeof(QueryVector));
    int node_idx = node_append(nodes, placeholder);

    if (len <= leaf_size) {
        int start = blocks->len;
        int block_count = (len + RINHA_LANES - 1) / RINHA_LANES;
        Reference zero;
        memset(&zero, 0, sizeof(zero));
        for (int b = 0; b < block_count; b++) {
            for (int lane = 0; lane < RINHA_LANES; lane++) {
                int i = b * RINHA_LANES + lane;
                ref_append(blocks, i < len ? refs[indices[i]] : zero);
            }
        }
        BuildNode *node = &nodes->items[node_idx];
        node->left = -1;
        node->right = -1;
        node->start = start;
        node->len = len;
        return node_idx;
    }

    int split_dim = widest_dimension(min, max);
    int *sorted = xmalloc((size_t)len * sizeof(int));
    memcpy(sorted, indices, (size_t)len * sizeof(int));
    sort_refs = refs;
    sort_dim = split_dim;
    qsort(sorted, (size_t)len, sizeof(int), cmp_index_by_dim);

    int left_len = len / 2;
    int left = build_tree(refs, sorted, left_len, leaf_size, blocks, nodes);
    int right = build_tree(refs, sorted + left_len, len - left_len, leaf_size, blocks, nodes);
    free(sorted);

    BuildNode left_node = nodes->items[left];
    BuildNode right_node = nodes->items[right];
    BuildNode *node = &nodes->items[node_idx];
    node->left = left;
    node->right = right;
    node->start = left_node.start;
    node->len = left_node.len + right_node.len;
    return node_idx;
}

static bool mkdir_parent(const char *path) {
    char tmp[PATH_MAX];
    size_t n = strlen(path);
    if (n >= sizeof(tmp)) return false;
    memcpy(tmp, path, n + 1);

    char *slash = strrchr(tmp, '/');
    if (slash == NULL) return true;
    *slash = '\0';
    if (tmp[0] == '\0') return true;

    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return false;
            *p = '/';
        }
    }
    return mkdir(tmp, 0755) == 0 || errno == EEXIST;
}

static void write_vector(uint8_t *buf, const QueryVector v) {
    for (int i = 0; i < RINHA_PACKED_DIMS; i++) {
        write_u16le(buf + i * 2, (uint16_t)v[i]);
    }
}

static void write_file(const char *path, const Reference *refs, int ref_count,
                       const PartRoot *roots, int root_count, const NodeVec *nodes, const RefVec *blocks) {
    if (!mkdir_parent(path)) {
        die("mkdir parent");
    }
    FILE *fp = fopen(path, "wb");
    if (fp == NULL) die("fopen output");

    int block_count = blocks->len / RINHA_LANES;
    uint8_t header[RINHA_HEADER_SIZE];
    memset(header, 0, sizeof(header));
    memcpy(header, RINHA_MAGIC, 8);
    write_u32le(header + 8, RINHA_SCALE);
    write_u32le(header + 12, RINHA_DIMS);
    write_u32le(header + 16, RINHA_PACKED_DIMS);
    write_u32le(header + 20, RINHA_LANES);
    write_u32le(header + 24, (uint32_t)ref_count);
    write_u32le(header + 28, (uint32_t)root_count);
    write_u32le(header + 32, (uint32_t)nodes->len);
    write_u32le(header + 36, (uint32_t)block_count);
    if (fwrite(header, 1, sizeof(header), fp) != sizeof(header)) die("write header");

    for (int i = 0; i < root_count; i++) {
        uint8_t buf[RINHA_PART_SIZE];
        memset(buf, 0, sizeof(buf));
        const BuildNode *root = &nodes->items[roots[i].root];
        write_u32le(buf, roots[i].key);
        write_u32le(buf + 4, (uint32_t)roots[i].root);
        write_u32le(buf + 8, (uint32_t)root->len);
        write_vector(buf + 12, root->min);
        write_vector(buf + 44, root->max);
        if (fwrite(buf, 1, sizeof(buf), fp) != sizeof(buf)) die("write partition");
    }

    for (int i = 0; i < nodes->len; i++) {
        uint8_t buf[RINHA_NODE_SIZE];
        memset(buf, 0, sizeof(buf));
        const BuildNode *n = &nodes->items[i];
        write_u32le(buf, (uint32_t)n->left);
        write_u32le(buf + 4, (uint32_t)n->right);
        write_u32le(buf + 8, (uint32_t)(n->start / RINHA_LANES));
        write_u32le(buf + 12, (uint32_t)n->len);
        write_vector(buf + 16, n->min);
        write_vector(buf + 48, n->max);
        if (fwrite(buf, 1, sizeof(buf), fp) != sizeof(buf)) die("write node");
    }

    size_t vector_count = (size_t)block_count * RINHA_DIMS * RINHA_LANES;
    int16_t *vectors = xmalloc(vector_count * sizeof(int16_t));
    uint8_t *labels = xmalloc((size_t)block_count * RINHA_LANES);
    for (int b = 0; b < block_count; b++) {
        for (int d = 0; d < RINHA_DIMS; d++) {
            for (int lane = 0; lane < RINHA_LANES; lane++) {
                vectors[(size_t)b * RINHA_DIMS * RINHA_LANES + d * RINHA_LANES + lane] =
                    blocks->items[b * RINHA_LANES + lane].vector[d];
            }
        }
        for (int lane = 0; lane < RINHA_LANES; lane++) {
            labels[b * RINHA_LANES + lane] = blocks->items[b * RINHA_LANES + lane].label;
        }
    }

    if (fwrite(vectors, sizeof(int16_t), vector_count, fp) != vector_count) die("write vectors");
    if (fwrite(labels, 1, (size_t)block_count * RINHA_LANES, fp) != (size_t)block_count * RINHA_LANES) die("write labels");
    free(vectors);
    free(labels);

    if (fclose(fp) != 0) die("close output");
    (void)refs;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <references.json.gz> <out.idx> [leaf-size]\n", argv[0]);
        return 2;
    }

    int leaf_size = 80;
    if (argc >= 4) {
        leaf_size = atoi(argv[3]);
    }
    if (leaf_size < 32) leaf_size = 32;
    if (leaf_size > 2048) leaf_size = 2048;

    RefVec refs = {0};
    parse_references(argv[1], &refs);
    if (refs.len == 0) {
        fprintf(stderr, "empty reference set\n");
        return 1;
    }

    IntVec partitions[256];
    memset(partitions, 0, sizeof(partitions));
    for (int i = 0; i < refs.len; i++) {
        uint32_t key = partition_key(refs.items[i].vector);
        int_append(&partitions[key], i);
    }

    NodeVec nodes = {0};
    RefVec blocks = {0};
    blocks.cap = refs.len + RINHA_LANES;
    blocks.items = xmalloc((size_t)blocks.cap * sizeof(Reference));

    PartRoot roots[256];
    int root_count = 0;
    for (int key = 0; key < 256; key++) {
        if (partitions[key].len == 0) continue;
        int root = build_tree(refs.items, partitions[key].items, partitions[key].len, leaf_size, &blocks, &nodes);
        roots[root_count].key = (uint32_t)key;
        roots[root_count].root = root;
        root_count++;
    }

    if (blocks.len % RINHA_LANES != 0) {
        fprintf(stderr, "internal block padding mismatch\n");
        return 1;
    }

    write_file(argv[2], refs.items, refs.len, roots, root_count, &nodes, &blocks);
    fprintf(stderr, "built refs=%d partitions=%d nodes=%d blocks=%d leaf=%d\n",
            refs.len, root_count, nodes.len, blocks.len / RINHA_LANES, leaf_size);

    for (int i = 0; i < 256; i++) free(partitions[i].items);
    free(refs.items);
    free(blocks.items);
    free(nodes.items);
    return 0;
}

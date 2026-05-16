#define _GNU_SOURCE

#include "index.h"
#include "payload.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#define RX_CAP 8192

typedef struct {
    const char *data;
    size_t len;
} Response;

#define RESPONSE(s) {s, sizeof(s) - 1}

static const Response fraud_responses[] = {
    RESPONSE("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}"),
    RESPONSE("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}"),
    RESPONSE("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}"),
    RESPONSE("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}"),
    RESPONSE("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}"),
    RESPONSE("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}"),
};

static const Response ready_response = RESPONSE("HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n");
static const Response not_found_response = RESPONSE("HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
static const Response bad_request_response = RESPONSE("HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");

typedef enum {
    PARSED_INCOMPLETE = 0,
    PARSED_BAD,
    PARSED_READY,
    PARSED_NOT_FOUND,
    PARSED_FRAUD,
} ParsedKind;

typedef struct {
    ParsedKind kind;
    int body_start;
    int body_end;
    int consumed;
} ParsedRequest;

static Index g_index;
static pthread_attr_t g_client_attr;

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

static int create_unix_listener(const char *path) {
    if (!mkdir_parent(path)) return -1;
    unlink(path);

    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) return -1;

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    if (strlen(path) >= sizeof(addr.sun_path)) {
        close(fd);
        return -1;
    }
    strcpy(addr.sun_path, path);

    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
    chmod(path, 0666);
    if (listen(fd, 1024) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static int recv_fd(int control_fd) {
    char one = 0;
    struct iovec iov = {.iov_base = &one, .iov_len = 1};
    char cmsg_buf[CMSG_SPACE(sizeof(int))];
    memset(cmsg_buf, 0, sizeof(cmsg_buf));
    struct msghdr msg;
    memset(&msg, 0, sizeof(msg));
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = cmsg_buf;
    msg.msg_controllen = sizeof(cmsg_buf);

    ssize_t n;
    do {
        n = recvmsg(control_fd, &msg, 0);
    } while (n < 0 && errno == EINTR);
    if (n <= 0) return -1;

    for (struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg); cmsg != NULL; cmsg = CMSG_NXTHDR(&msg, cmsg)) {
        if (cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS && cmsg->cmsg_len >= CMSG_LEN(sizeof(int))) {
            int fd;
            memcpy(&fd, CMSG_DATA(cmsg), sizeof(fd));
            return fd;
        }
    }
    return -1;
}

static void set_client_socket_options(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags >= 0) {
        fcntl(fd, F_SETFL, flags & ~O_NONBLOCK);
    }
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

static ssize_t read_fd(int fd, uint8_t *buf, size_t len) {
    for (;;) {
        ssize_t n = read(fd, buf, len);
        if (n < 0 && errno == EINTR) continue;
        return n;
    }
}

static bool write_all(int fd, const char *buf, size_t len) {
    while (len > 0) {
        ssize_t n = write(fd, buf, len);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        buf += n;
        len -= (size_t)n;
    }
    return true;
}

static int find_header_end(const uint8_t *buf, int len) {
    for (int i = 0; i + 3 < len; i++) {
        if (buf[i] == '\r' && buf[i + 1] == '\n' && buf[i + 2] == '\r' && buf[i + 3] == '\n') {
            return i;
        }
    }
    return -1;
}

static int find_byte(const uint8_t *buf, int len, uint8_t needle) {
    for (int i = 0; i < len; i++) {
        if (buf[i] == needle) return i;
    }
    return -1;
}

static bool path_eq(const uint8_t *rest, int rest_len, const char *path) {
    int path_len = (int)strlen(path);
    if (rest_len < path_len + 1) return false;
    if (memcmp(rest, path, (size_t)path_len) != 0) return false;
    uint8_t next = rest[path_len];
    return next == ' ' || next == '?';
}

static int parse_content_length(const uint8_t *headers, int len) {
    static const char name[] = "content-length:";
    const int name_len = (int)sizeof(name) - 1;
    for (int i = 0; i + name_len <= len; i++) {
        uint8_t c = headers[i];
        if (c != 'c' && c != 'C') continue;
        bool match = true;
        for (int j = 1; j < name_len; j++) {
            if ((headers[i + j] | 0x20) != (uint8_t)name[j]) {
                match = false;
                break;
            }
        }
        if (!match) continue;

        int p = i + name_len;
        while (p < len && (headers[p] == ' ' || headers[p] == '\t')) p++;
        int value = 0;
        while (p < len && headers[p] >= '0' && headers[p] <= '9') {
            value = value * 10 + (int)(headers[p] - '0');
            p++;
        }
        return value;
    }
    return 0;
}

static ParsedRequest parse_request(const uint8_t *buf, int len) {
    ParsedRequest out = {.kind = PARSED_INCOMPLETE};
    if (len < 16) return out;

    int header_end = find_header_end(buf, len);
    if (header_end < 0) return out;

    int line_end = find_byte(buf, header_end, '\r');
    if (line_end < 0) {
        out.kind = PARSED_BAD;
        return out;
    }

    if (line_end >= 5 && memcmp(buf, "POST ", 5) == 0) {
        const uint8_t *rest = buf + 5;
        int rest_len = line_end - 5;
        if (path_eq(rest, rest_len, "/fraud-score")) {
            int content_length = parse_content_length(buf + line_end, header_end - line_end);
            int body_start = header_end + 4;
            int body_end = body_start + content_length;
            if (len < body_end) return out;
            out.kind = PARSED_FRAUD;
            out.body_start = body_start;
            out.body_end = body_end;
            out.consumed = body_end;
            return out;
        }
        out.kind = PARSED_NOT_FOUND;
        out.consumed = header_end + 4;
        return out;
    }

    if (line_end >= 4 && memcmp(buf, "GET ", 4) == 0) {
        const uint8_t *rest = buf + 4;
        int rest_len = line_end - 4;
        out.kind = path_eq(rest, rest_len, "/ready") ? PARSED_READY : PARSED_NOT_FOUND;
        out.consumed = header_end + 4;
        return out;
    }

    out.kind = PARSED_BAD;
    return out;
}

static Response handle_fraud(const uint8_t *body, size_t len) {
    Payload payload;
    if (!payload_parse(body, len, &payload)) {
        return fraud_responses[0];
    }

    QueryVector q;
    payload_to_vector(&payload, q);
    uint8_t count = index_predict_fraud_count(&g_index, q);
    if (count > 5) count = 5;
    return fraud_responses[count];
}

static void *serve_client(void *arg) {
    int fd = (int)(intptr_t)arg;
    uint8_t rx[RX_CAP];
    int head = 0;
    int tail = 0;

    for (;;) {
        while (head < tail) {
            ParsedRequest req = parse_request(rx + head, tail - head);
            if (req.kind == PARSED_INCOMPLETE) break;
            if (req.kind == PARSED_BAD) {
                write_all(fd, bad_request_response.data, bad_request_response.len);
                close(fd);
                return NULL;
            }

            Response response;
            if (req.kind == PARSED_READY) {
                response = ready_response;
            } else if (req.kind == PARSED_NOT_FOUND) {
                response = not_found_response;
            } else {
                response = handle_fraud(rx + head + req.body_start, (size_t)(req.body_end - req.body_start));
            }

            if (!write_all(fd, response.data, response.len)) {
                close(fd);
                return NULL;
            }
            head += req.consumed;
        }

        if (head == tail) {
            head = 0;
            tail = 0;
        } else if (tail == RX_CAP && head > 0) {
            memmove(rx, rx + head, (size_t)(tail - head));
            tail -= head;
            head = 0;
        }
        if (tail == RX_CAP) {
            close(fd);
            return NULL;
        }

        ssize_t n = read_fd(fd, rx + tail, (size_t)(RX_CAP - tail));
        if (n <= 0) {
            close(fd);
            return NULL;
        }
        tail += (int)n;
    }
}

static void spawn_client(int fd) {
    set_client_socket_options(fd);
    pthread_t tid;
    if (pthread_create(&tid, &g_client_attr, serve_client, (void *)(intptr_t)fd) != 0) {
        close(fd);
        return;
    }
    pthread_detach(tid);
}

static void *serve_control(void *arg) {
    int control_fd = (int)(intptr_t)arg;
    for (;;) {
        int fd = recv_fd(control_fd);
        if (fd < 0) break;
        spawn_client(fd);
    }
    close(control_fd);
    return NULL;
}

static const char *getenv_default(const char *name, const char *fallback) {
    const char *value = getenv(name);
    return (value != NULL && value[0] != '\0') ? value : fallback;
}

int main(void) {
    signal(SIGPIPE, SIG_IGN);

    const char *sock_path = getenv_default("FD_SOCKET", getenv_default("RINHA_FD_SOCKET", ""));
    const char *index_path = getenv_default("INDEX_PATH", getenv_default("RINHA_INDEX_PATH", ""));
    if (sock_path[0] == '\0' || index_path[0] == '\0') {
        fprintf(stderr, "FD_SOCKET and INDEX_PATH are required\n");
        return 2;
    }

    if (!index_open(&g_index, index_path)) {
        fprintf(stderr, "failed to open index %s\n", index_path);
        return 1;
    }

    pthread_attr_init(&g_client_attr);
    size_t stack_size = 64 * 1024;
    if (stack_size < (size_t)PTHREAD_STACK_MIN) stack_size = (size_t)PTHREAD_STACK_MIN;
    pthread_attr_setstacksize(&g_client_attr, stack_size);

    int listener = create_unix_listener(sock_path);
    if (listener < 0) {
        fprintf(stderr, "failed to listen on %s: %s\n", sock_path, strerror(errno));
        return 1;
    }

    fprintf(stderr, "ready sock=%s index=%s\n", sock_path, index_path);
    for (;;) {
        int control_fd = accept4(listener, NULL, NULL, SOCK_CLOEXEC);
        if (control_fd < 0) {
            if (errno == EINTR) continue;
            continue;
        }
        pthread_t tid;
        if (pthread_create(&tid, NULL, serve_control, (void *)(intptr_t)control_fd) != 0) {
            close(control_fd);
            continue;
        }
        pthread_detach(tid);
    }
}

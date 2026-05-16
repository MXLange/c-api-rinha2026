#define _GNU_SOURCE

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#define MAX_UPSTREAMS 16

typedef struct {
    const char *path;
    int fd;
} Sender;

typedef struct {
    int listener_fd;
    char *upstreams[MAX_UPSTREAMS];
    int upstream_count;
    atomic_ulong next;
} LoadBalancer;

typedef struct {
    LoadBalancer *lb;
    Sender senders[MAX_UPSTREAMS];
} Worker;

static const char bad_gateway[] = "HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";

static int getenv_int(const char *name, int fallback) {
    const char *value = getenv(name);
    if (value == NULL || value[0] == '\0') return fallback;
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    if (end == value || parsed < 1 || parsed > 65535) return fallback;
    return (int)parsed;
}

static void parse_upstreams(LoadBalancer *lb, const char *value) {
    if (value == NULL || value[0] == '\0') {
        lb->upstreams[0] = strdup("/run/sock/api1.sock");
        lb->upstreams[1] = strdup("/run/sock/api2.sock");
        lb->upstream_count = 2;
        return;
    }

    char *copy = strdup(value);
    if (copy == NULL) abort();
    char *save = NULL;
    for (char *part = strtok_r(copy, ",", &save);
         part != NULL && lb->upstream_count < MAX_UPSTREAMS;
         part = strtok_r(NULL, ",", &save)) {
        while (*part == ' ' || *part == '\t') part++;
        char *end = part + strlen(part);
        while (end > part && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\n' || end[-1] == '\r')) {
            *--end = '\0';
        }
        if (*part != '\0') {
            lb->upstreams[lb->upstream_count++] = strdup(part);
        }
    }
    free(copy);
}

static int parse_port(void) {
    const char *bind_addr = getenv("BIND_ADDR");
    if (bind_addr != NULL && bind_addr[0] != '\0') {
        const char *colon = strrchr(bind_addr, ':');
        if (colon != NULL && colon[1] != '\0') {
            return atoi(colon + 1);
        }
        return atoi(bind_addr);
    }
    return getenv_int("PORT", 9999);
}

static int create_tcp_listener(int port) {
    int fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) return -1;

    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#ifdef SO_REUSEPORT
    setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &one, sizeof(one));
#endif

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((uint16_t)port);
    if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
    if (listen(fd, 8192) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static int connect_unix(const char *path) {
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

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static void wait_upstreams(LoadBalancer *lb) {
    struct timespec delay = {.tv_sec = 0, .tv_nsec = 50 * 1000 * 1000};
    for (int tries = 0; tries < 100; tries++) {
        int ready = 0;
        for (int i = 0; i < lb->upstream_count; i++) {
            int fd = connect_unix(lb->upstreams[i]);
            if (fd >= 0) {
                ready++;
                close(fd);
            }
        }
        if (ready == lb->upstream_count) return;
        nanosleep(&delay, NULL);
    }
}

static bool send_fd_once(int control_fd, int client_fd) {
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

    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cmsg), &client_fd, sizeof(client_fd));
    msg.msg_controllen = cmsg->cmsg_len;

    ssize_t n;
    do {
        n = sendmsg(control_fd, &msg, MSG_NOSIGNAL);
    } while (n < 0 && errno == EINTR);
    return n == 1;
}

static bool sender_send(Sender *sender, int client_fd) {
    if (sender->fd < 0) {
        sender->fd = connect_unix(sender->path);
        if (sender->fd < 0) return false;
    }
    if (send_fd_once(sender->fd, client_fd)) return true;

    close(sender->fd);
    sender->fd = connect_unix(sender->path);
    if (sender->fd < 0) return false;
    return send_fd_once(sender->fd, client_fd);
}

static void set_client_options(int fd) {
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

static bool pass_fd(Worker *worker, int client_fd) {
    LoadBalancer *lb = worker->lb;
    unsigned long start = atomic_fetch_add_explicit(&lb->next, 1, memory_order_relaxed);
    for (int i = 0; i < lb->upstream_count; i++) {
        int idx = (int)((start + (unsigned long)i) % (unsigned long)lb->upstream_count);
        if (sender_send(&worker->senders[idx], client_fd)) {
            return true;
        }
    }
    return false;
}

static void write_bad_gateway(int fd) {
    const char *p = bad_gateway;
    size_t n = sizeof(bad_gateway) - 1;
    while (n > 0) {
        ssize_t written = write(fd, p, n);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) return;
        p += written;
        n -= (size_t)written;
    }
}

static void *worker_loop(void *arg) {
    Worker *worker = arg;
    for (;;) {
        int client_fd = accept4(worker->lb->listener_fd, NULL, NULL, SOCK_CLOEXEC);
        if (client_fd < 0) {
            if (errno == EINTR) continue;
            continue;
        }
        set_client_options(client_fd);
        if (!pass_fd(worker, client_fd)) {
            write_bad_gateway(client_fd);
        }
        close(client_fd);
    }
    return NULL;
}

int main(void) {
    signal(SIGPIPE, SIG_IGN);

    LoadBalancer lb;
    memset(&lb, 0, sizeof(lb));
    parse_upstreams(&lb, getenv("FD_UPSTREAMS"));
    if (lb.upstream_count == 0) {
        fprintf(stderr, "no upstreams configured\n");
        return 2;
    }

    int port = parse_port();
    int workers = getenv_int("WORKERS", 2);
    lb.listener_fd = create_tcp_listener(port);
    if (lb.listener_fd < 0) {
        fprintf(stderr, "listen port %d failed: %s\n", port, strerror(errno));
        return 1;
    }

    wait_upstreams(&lb);
    fprintf(stderr, "ready port=%d workers=%d upstreams=%d\n", port, workers, lb.upstream_count);

    Worker *worker_items = calloc((size_t)workers, sizeof(Worker));
    pthread_t *threads = calloc((size_t)workers, sizeof(pthread_t));
    if (worker_items == NULL || threads == NULL) abort();

    for (int i = 0; i < workers; i++) {
        worker_items[i].lb = &lb;
        for (int j = 0; j < lb.upstream_count; j++) {
            worker_items[i].senders[j].path = lb.upstreams[j];
            worker_items[i].senders[j].fd = -1;
        }
        if (pthread_create(&threads[i], NULL, worker_loop, &worker_items[i]) != 0) {
            fprintf(stderr, "worker create failed\n");
            return 1;
        }
    }

    for (int i = 0; i < workers; i++) {
        pthread_join(threads[i], NULL);
    }
    return 0;
}

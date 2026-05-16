CC ?= gcc
CFLAGS ?= -O3 -march=haswell -mtune=haswell -mavx2 -flto -fno-plt -DNDEBUG -std=c11 -Wall -Wextra -Wshadow
LDFLAGS ?= -flto

BUILD_DIR := build
API_OBJS := $(BUILD_DIR)/api.o $(BUILD_DIR)/index.o $(BUILD_DIR)/payload.o
LB_OBJS := $(BUILD_DIR)/lb.o
BUILD_INDEX_OBJS := $(BUILD_DIR)/build_index.o $(BUILD_DIR)/payload.o

.PHONY: all clean strip

all: api lb build-index

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: src/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

api: $(API_OBJS)
	$(CC) $(CFLAGS) $(API_OBJS) $(LDFLAGS) -pthread -lm -o $@

lb: $(LB_OBJS)
	$(CC) $(CFLAGS) $(LB_OBJS) $(LDFLAGS) -pthread -o $@

build-index: $(BUILD_INDEX_OBJS)
	$(CC) $(CFLAGS) $(BUILD_INDEX_OBJS) $(LDFLAGS) -lm -o $@

strip: all
	strip -s api lb build-index

clean:
	rm -rf $(BUILD_DIR) api lb build-index

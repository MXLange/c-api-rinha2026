#ifndef RINHA_PAYLOAD_H
#define RINHA_PAYLOAD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "common.h"

typedef struct {
    uint32_t amount_milli;
    uint32_t customer_avg_amount_milli;
    uint32_t merchant_avg_amount_milli;
    uint32_t km_from_home_milli;
    uint32_t km_from_current_milli;
    uint32_t tx_count_24h;
    uint32_t mcc;
    uint32_t minutes_since_last;
    uint8_t installments;
    uint8_t hour;
    uint8_t day_of_week;
    bool is_online;
    bool card_present;
    bool is_unknown_merchant;
    bool has_last_tx;
} Payload;

bool payload_parse(const uint8_t *buf, size_t len, Payload *out);
void payload_to_vector(const Payload *p, QueryVector out);
int16_t quantize_value(double value);

#endif

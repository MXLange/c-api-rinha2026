#include "payload.h"

#include <math.h>
#include <string.h>

typedef struct {
    const uint8_t *ptr;
    size_t len;
} Slice;

static inline bool is_digit(uint8_t b) {
    return b >= '0' && b <= '9';
}

static bool next_value(size_t *pos, const uint8_t *buf, size_t len) {
    while (*pos < len) {
        uint8_t c = buf[*pos];
        if (c == ':') {
            (*pos)++;
            while (*pos < len) {
                c = buf[*pos];
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                    (*pos)++;
                    continue;
                }
                return true;
            }
            return false;
        }
        if (c == '"') {
            (*pos)++;
            while (*pos < len && buf[*pos] != '"') {
                (*pos)++;
            }
            if (*pos >= len) return false;
            (*pos)++;
            continue;
        }
        (*pos)++;
    }
    return false;
}

static bool skip_string(size_t *pos, const uint8_t *buf, size_t len) {
    if (*pos < len && buf[*pos] == '"') {
        (*pos)++;
    }
    while (*pos < len && buf[*pos] != '"') {
        (*pos)++;
    }
    if (*pos >= len) return false;
    (*pos)++;
    return true;
}

static uint32_t parse_scaled1000(const uint8_t *buf, size_t len, size_t *used) {
    size_t pos = 0;
    bool negative = false;
    if (pos < len && buf[pos] == '-') {
        negative = true;
        pos++;
    }

    uint64_t int_part = 0;
    while (pos < len && is_digit(buf[pos])) {
        int_part = int_part * 10 + (uint64_t)(buf[pos] - '0');
        pos++;
    }

    uint64_t value = int_part * 1000u;
    if (pos < len && buf[pos] == '.') {
        pos++;
        uint32_t frac = 0;
        int digits = 0;
        int round_digit = -1;
        while (pos < len && is_digit(buf[pos])) {
            if (digits < 3) {
                frac = frac * 10u + (uint32_t)(buf[pos] - '0');
            } else if (digits == 3) {
                round_digit = (int)(buf[pos] - '0');
            }
            digits++;
            pos++;
        }
        while (digits < 3) {
            frac *= 10u;
            digits++;
        }
        value += frac;
        if (round_digit >= 5) {
            value++;
        }
    }

    if (pos < len && (buf[pos] == 'e' || buf[pos] == 'E')) {
        pos++;
        int sign = 1;
        if (pos < len && (buf[pos] == '+' || buf[pos] == '-')) {
            if (buf[pos] == '-') sign = -1;
            pos++;
        }
        int exp = 0;
        while (pos < len && is_digit(buf[pos])) {
            exp = exp * 10 + (int)(buf[pos] - '0');
            pos++;
        }
        if (sign > 0) {
            while (exp-- > 0 && value <= UINT64_MAX / 10u) {
                value *= 10u;
            }
        } else {
            while (exp-- > 0) {
                value = (value + 5u) / 10u;
            }
        }
    }

    *used = pos;
    if (negative) return 0;
    if (value > UINT32_MAX) return UINT32_MAX;
    return (uint32_t)value;
}

static uint32_t scan_scaled1000(size_t *pos, const uint8_t *buf, size_t len) {
    size_t used = 0;
    uint32_t value = parse_scaled1000(buf + *pos, len - *pos, &used);
    *pos += used;
    return value;
}

static uint32_t scan_uint32(size_t *pos, const uint8_t *buf, size_t len) {
    uint32_t value = 0;
    while (*pos < len && is_digit(buf[*pos])) {
        value = value * 10u + (uint32_t)(buf[*pos] - '0');
        (*pos)++;
    }
    return value;
}

static bool scan_bool(size_t *pos, const uint8_t *buf, size_t len) {
    bool is_true = *pos < len && buf[*pos] == 't';
    *pos += is_true ? 4u : 5u;
    return is_true;
}

static bool scan_string(size_t *pos, const uint8_t *buf, size_t len, Slice *out) {
    if (*pos >= len || buf[*pos] != '"') return false;
    (*pos)++;
    size_t start = *pos;
    while (*pos < len && buf[*pos] != '"') {
        (*pos)++;
    }
    if (*pos >= len) return false;
    out->ptr = buf + start;
    out->len = *pos - start;
    (*pos)++;
    return true;
}

static uint32_t scan_mcc(size_t *pos, const uint8_t *buf, size_t len) {
    if (*pos < len && buf[*pos] == '"') {
        (*pos)++;
    }
    uint32_t value = scan_uint32(pos, buf, len);
    if (*pos < len && buf[*pos] == '"') {
        (*pos)++;
    }
    return value;
}

static bool scan_iso(size_t *pos, const uint8_t *buf, size_t len, uint16_t *year, uint8_t *month, uint8_t *day, uint8_t *hour, uint8_t *minute) {
    if (*pos < len && buf[*pos] == '"') {
        (*pos)++;
    }
    if (len - *pos < 20) return false;
    const uint8_t *s = buf + *pos;
    *year = (uint16_t)((s[0] - '0') * 1000 + (s[1] - '0') * 100 + (s[2] - '0') * 10 + (s[3] - '0'));
    *month = (uint8_t)((s[5] - '0') * 10 + (s[6] - '0'));
    *day = (uint8_t)((s[8] - '0') * 10 + (s[9] - '0'));
    *hour = (uint8_t)((s[11] - '0') * 10 + (s[12] - '0'));
    *minute = (uint8_t)((s[14] - '0') * 10 + (s[15] - '0'));
    *pos += 20;
    while (*pos < len && buf[*pos] != '"') {
        (*pos)++;
    }
    if (*pos < len) {
        (*pos)++;
    }
    return true;
}

static uint8_t day_of_week(uint16_t year, uint8_t month, uint8_t day) {
    static const uint16_t t[] = {0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4};
    uint32_t y = year;
    if (month < 3) y--;
    uint32_t dow = (y + y / 4 - y / 100 + y / 400 + t[month - 1] + day) % 7;
    return (uint8_t)((dow + 6) % 7);
}

static int64_t days_since_epoch(int32_t year, uint32_t month, uint32_t day) {
    int32_t y = year;
    if (month <= 2) y--;
    int32_t era = y / 400;
    if (y < 0 && y % 400 != 0) era--;
    uint32_t yoe = (uint32_t)(y - era * 400);
    uint32_t m = month;
    if (m > 2) {
        m -= 3;
    } else {
        m += 9;
    }
    uint32_t doy = (153 * m + 2) / 5 + day - 1;
    uint32_t doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return (int64_t)era * 146097 + (int64_t)doe - 719468;
}

static uint32_t minutes_between(uint16_t y1, uint8_t mo1, uint8_t d1, uint8_t h1, uint8_t mi1,
                                uint16_t y2, uint8_t mo2, uint8_t d2, uint8_t h2, uint8_t mi2) {
    int64_t day1 = days_since_epoch((int32_t)y1, mo1, d1);
    int64_t day2 = days_since_epoch((int32_t)y2, mo2, d2);
    int64_t min1 = day1 * 1440 + (int64_t)h1 * 60 + mi1;
    int64_t min2 = day2 * 1440 + (int64_t)h2 * 60 + mi2;
    if (min2 <= min1) return 0;
    return (uint32_t)(min2 - min1);
}

static bool slices_equal(Slice a, Slice b) {
    return a.len == b.len && memcmp(a.ptr, b.ptr, a.len) == 0;
}

bool payload_parse(const uint8_t *buf, size_t len, Payload *out) {
    Payload p;
    memset(&p, 0, sizeof(p));
    size_t pos = 0;

    if (!next_value(&pos, buf, len)) return false;
    if (!skip_string(&pos, buf, len)) return false;

    if (!next_value(&pos, buf, len) || !next_value(&pos, buf, len)) return false;
    p.amount_milli = scan_scaled1000(&pos, buf, len);

    if (!next_value(&pos, buf, len)) return false;
    p.installments = (uint8_t)scan_uint32(&pos, buf, len);

    if (!next_value(&pos, buf, len)) return false;
    uint16_t req_y;
    uint8_t req_mo, req_d, req_h, req_min;
    if (!scan_iso(&pos, buf, len, &req_y, &req_mo, &req_d, &req_h, &req_min)) return false;
    p.hour = req_h;
    p.day_of_week = day_of_week(req_y, req_mo, req_d);

    if (!next_value(&pos, buf, len) || !next_value(&pos, buf, len)) return false;
    p.customer_avg_amount_milli = scan_scaled1000(&pos, buf, len);

    if (!next_value(&pos, buf, len)) return false;
    p.tx_count_24h = scan_uint32(&pos, buf, len);

    if (!next_value(&pos, buf, len) || pos >= len) return false;
    pos++;
    Slice merchants[16];
    int merchant_count = 0;
    while (pos < len && buf[pos] != ']') {
        if (buf[pos] == '"') {
            pos++;
            size_t start = pos;
            while (pos < len && buf[pos] != '"') {
                pos++;
            }
            if (pos >= len) return false;
            if (merchant_count < 16) {
                merchants[merchant_count].ptr = buf + start;
                merchants[merchant_count].len = pos - start;
                merchant_count++;
            }
            pos++;
            continue;
        }
        pos++;
    }
    if (pos < len) pos++;

    if (!next_value(&pos, buf, len) || !next_value(&pos, buf, len)) return false;
    Slice merchant_id;
    if (!scan_string(&pos, buf, len, &merchant_id)) return false;

    if (!next_value(&pos, buf, len)) return false;
    p.mcc = scan_mcc(&pos, buf, len);

    if (!next_value(&pos, buf, len)) return false;
    p.merchant_avg_amount_milli = scan_scaled1000(&pos, buf, len);

    if (!next_value(&pos, buf, len) || !next_value(&pos, buf, len)) return false;
    p.is_online = scan_bool(&pos, buf, len);

    if (!next_value(&pos, buf, len)) return false;
    p.card_present = scan_bool(&pos, buf, len);

    if (!next_value(&pos, buf, len)) return false;
    p.km_from_home_milli = scan_scaled1000(&pos, buf, len);

    if (!next_value(&pos, buf, len) || pos >= len) return false;
    p.has_last_tx = buf[pos] != 'n';
    if (p.has_last_tx) {
        if (!next_value(&pos, buf, len)) return false;
        uint16_t last_y;
        uint8_t last_mo, last_d, last_h, last_min;
        if (!scan_iso(&pos, buf, len, &last_y, &last_mo, &last_d, &last_h, &last_min)) return false;
        if (!next_value(&pos, buf, len)) return false;
        p.km_from_current_milli = scan_scaled1000(&pos, buf, len);
        p.minutes_since_last = minutes_between(last_y, last_mo, last_d, last_h, last_min,
                                               req_y, req_mo, req_d, req_h, req_min);
    }

    p.is_unknown_merchant = true;
    for (int i = 0; i < merchant_count; i++) {
        if (slices_equal(merchants[i], merchant_id)) {
            p.is_unknown_merchant = false;
            break;
        }
    }

    *out = p;
    return true;
}

int16_t quantize_value(double value) {
    if (value <= -1.0) return -RINHA_SCALE;
    if (value <= 0.0) return 0;
    if (value >= 1.0) return RINHA_SCALE;
    return (int16_t)llround(value * (double)RINHA_SCALE);
}

static int16_t mcc_risk_quantized(uint32_t mcc) {
    switch (mcc) {
        case 5411: return 1500;
        case 5812: return 3000;
        case 5912: return 2000;
        case 5944: return 4500;
        case 7801: return 8000;
        case 7802: return 7500;
        case 7995: return 8500;
        case 4511: return 3500;
        case 5311: return 2500;
        case 5999: return 5000;
        default: return 5000;
    }
}

static inline uint64_t div_round_u64(uint64_t numerator, uint64_t denominator) {
    return (numerator + denominator / 2u) / denominator;
}

static inline int16_t clamp_quant_u64(uint64_t value) {
    return value >= RINHA_SCALE ? RINHA_SCALE : (int16_t)value;
}

static inline int16_t quantize_uint_div(uint32_t value, uint32_t denominator) {
    return clamp_quant_u64(div_round_u64((uint64_t)value * RINHA_SCALE, denominator));
}

static inline int16_t quantize_milli_div(uint32_t value_milli, uint32_t denominator_units) {
    return clamp_quant_u64(div_round_u64((uint64_t)value_milli * RINHA_SCALE, (uint64_t)denominator_units * 1000u));
}

static inline int16_t quantize_amount_ratio(uint32_t amount_milli, uint32_t avg_milli) {
    if (avg_milli == 0) return RINHA_SCALE;
    return clamp_quant_u64(div_round_u64((uint64_t)amount_milli * 1000u, avg_milli));
}

void payload_to_vector(const Payload *p, QueryVector out) {
    memset(out, 0, sizeof(QueryVector));
    out[0] = quantize_milli_div(p->amount_milli, 10000);
    out[1] = quantize_uint_div(p->installments, 12);
    out[2] = quantize_amount_ratio(p->amount_milli, p->customer_avg_amount_milli);
    out[3] = quantize_uint_div(p->hour, 23);
    out[4] = quantize_uint_div(p->day_of_week, 6);
    if (p->has_last_tx) {
        out[5] = quantize_uint_div(p->minutes_since_last, 1440);
        out[6] = quantize_milli_div(p->km_from_current_milli, 1000);
    } else {
        out[5] = -RINHA_SCALE;
        out[6] = -RINHA_SCALE;
    }
    out[7] = quantize_milli_div(p->km_from_home_milli, 1000);
    out[8] = quantize_uint_div(p->tx_count_24h, 20);
    if (p->is_online) out[9] = RINHA_SCALE;
    if (p->card_present) out[10] = RINHA_SCALE;
    if (p->is_unknown_merchant) out[11] = RINHA_SCALE;
    out[12] = mcc_risk_quantized(p->mcc);
    out[13] = quantize_milli_div(p->merchant_avg_amount_milli, 10000);
}

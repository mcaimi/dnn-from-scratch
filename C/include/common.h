#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <float.h>
#include <time.h>

#define TRUE 1
#define FALSE 0

static union { uint8_t test[4]; uint32_t mem_repr; } endiannes_test = { {0xCA, 0x00, 0x00, 0xFE} };
#define IS_LE ((uint8_t)endiannes_test.mem_repr == 0xCA)
#define IS_BE ((uint8_t)endiannes_test.mem_repr == 0xFE)


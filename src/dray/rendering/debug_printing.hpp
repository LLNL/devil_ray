#ifndef DRAY_DEBUG_PRINTF_HPP
#define DRAY_DEBUG_PRINTF_HPP

//#define DEBUG_PRINTF 1

#ifdef DEBUG_PRINTF
#define kernel_printf(...) printf(__VA_ARGS__)
#else
#define kernel_printf(...)
#endif

#endif

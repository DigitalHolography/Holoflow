#pragma once

#include <cstdlib>
#include <spdlog/spdlog.h>

#ifndef DH_ORG
#define DH_ORG "Digital Holography"
#endif

#ifndef DH_CONTACT
#define DH_CONTACT "guilloujules@gmail.com"
#endif

// ANSI escape codes for red text and reset
#define ANSI_RED "\033[31m"
#define ANSI_RESET "\033[0m"

#define DH_BUG(fmt, ...)                                                       \
  do {                                                                         \
    spdlog::critical(ANSI_RED                                                  \
                     "THIS IS A BUG!\nPlease report it to [{}] at [{}].\n" fmt \
                     "\n(File: {}, Line: {})" ANSI_RESET,                      \
                     DH_ORG, DH_CONTACT, ##__VA_ARGS__, __FILE__, __LINE__);   \
    std::abort();                                                              \
  } while (0)

#define DH_CHECK(condition, ...)                                               \
  do {                                                                         \
    if (!(condition)) {                                                        \
      DH_BUG("Check failed: {}." __VA_OPT__(" ") __VA_ARGS__, #condition);     \
    }                                                                          \
  } while (0)

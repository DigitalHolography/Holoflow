// Copyright 2025 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <spdlog/spdlog.h>

// TODO: Move to a config file
#define DH_ORG "Digital Holography"
#define DH_CONTACT "guilloujules@gmail.com"

// TODO: Move these to a better place
#define ANSI_RED "\033[31m"
#define ANSI_RESET "\033[0m"

#define HOLOFLOW_BUG(fmt, ...)                                                 \
  do {                                                                         \
    spdlog::critical(ANSI_RED                                                  \
                     "THIS IS A BUG!\nPlease report it to [{}] at [{}].\n" fmt \
                     "\n(File: {}, Line: {})" ANSI_RESET,                      \
                     DH_ORG, DH_CONTACT, ##__VA_ARGS__, __FILE__, __LINE__);   \
    std::abort();                                                              \
  } while (0)

#define HOLOFLOW_CHECK(condition, ...)                                         \
  do {                                                                         \
    if (!(condition)) {                                                        \
      HOLOFLOW_BUG("Check failed: {}." __VA_OPT__(" ") __VA_ARGS__,            \
                   #condition);                                                \
    }                                                                          \
  } while (0)

#define HOLOFLOW_UNIMPLEMENTED() HOLOFLOW_BUG("Unimplemented code reached.")

#define HOLOFLOW_UNREACHABLE() HOLOFLOW_BUG("Unreachable code reached.")
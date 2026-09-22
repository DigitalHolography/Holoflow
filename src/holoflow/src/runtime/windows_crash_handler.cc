// Copyright 2026 Digital Holography Foundation
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

#include "holoflow/runtime/windows_crash_handler.hh"

#include <atomic>
#include <utility>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

namespace holoflow::runtime {
namespace {

#if defined(_WIN32)
WindowsCrashHandler::Callback          callback;
std::atomic_flag                       callback_invoked = ATOMIC_FLAG_INIT;
LPTOP_LEVEL_EXCEPTION_FILTER           previous_filter = nullptr;
bool                                    installed       = false;

LONG WINAPI handle_unhandled_exception(EXCEPTION_POINTERS *exception_pointers) noexcept {
  if (callback && !callback_invoked.test_and_set(std::memory_order_acq_rel)) {
    unsigned long exception_code = 0;
    if (exception_pointers != nullptr && exception_pointers->ExceptionRecord != nullptr) {
      exception_code = exception_pointers->ExceptionRecord->ExceptionCode;
    }

    try {
      callback(exception_code);
    } catch (...) {
      // A crash callback must never interfere with the original exception.
    }
  }

  return EXCEPTION_CONTINUE_SEARCH;
}
#endif

} // namespace

void WindowsCrashHandler::install(Callback new_callback) {
#if defined(_WIN32)
  callback_invoked.clear(std::memory_order_release);
  callback = std::move(new_callback);
  if (!installed) {
    previous_filter = SetUnhandledExceptionFilter(&handle_unhandled_exception);
    installed       = true;
  }
#else
  (void)new_callback;
#endif
}

void WindowsCrashHandler::uninstall() {
#if defined(_WIN32)
  if (!installed) {
    return;
  }
  SetUnhandledExceptionFilter(previous_filter);
  previous_filter = nullptr;
  callback        = {};
  installed       = false;
#endif
}

} // namespace holoflow::runtime

// Copyright 2026 Digital Holography Foundation

#include <fstream>
#include <string>

#include "holoflow/runtime/windows_crash_handler.hh"

int main(int argc, char **argv) {
  if (argc != 2) {
    return 2;
  }

  holoflow::runtime::WindowsCrashHandler::install(
      [path = std::string{argv[1]}](unsigned long exception_code) {
    std::ofstream output(path, std::ios::trunc);
    output << std::hex << exception_code;
    output.flush();
      });

  volatile int *null_pointer = nullptr;
  *null_pointer = 42;
  return 0;
}

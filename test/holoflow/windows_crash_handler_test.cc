// Copyright 2026 Digital Holography Foundation

#include <gtest/gtest.h>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

TEST(WindowsCrashHandlerTest, WritesArtifactBeforeAccessViolationTerminatesChild) {
  wchar_t module_path[MAX_PATH]{};
  ASSERT_GT(GetModuleFileNameW(nullptr, module_path, MAX_PATH), 0U);

  const auto helper_path =
      std::filesystem::path(module_path).parent_path() / "windows_crash_handler_helper.exe";
  ASSERT_TRUE(std::filesystem::exists(helper_path)) << helper_path.string();

  const auto artifact = std::filesystem::temp_directory_path() /
                        ("holoflow_windows_crash_" + std::to_string(GetCurrentProcessId()) +
                         ".txt");
  std::error_code error;
  std::filesystem::remove(artifact, error);

  std::wstring command_line = L"\"" + helper_path.wstring() + L"\" \"" + artifact.wstring() +
                              L"\"";
  std::vector<wchar_t> command_buffer(command_line.begin(), command_line.end());
  command_buffer.push_back(L'\0');

  STARTUPINFOW startup_info{};
  startup_info.cb = sizeof(startup_info);
  PROCESS_INFORMATION process_info{};
  ASSERT_TRUE(CreateProcessW(nullptr, command_buffer.data(), nullptr, nullptr, FALSE, 0, nullptr,
                             nullptr, &startup_info, &process_info));

  ASSERT_EQ(WaitForSingleObject(process_info.hProcess, 5000), WAIT_OBJECT_0);
  DWORD exit_code = STILL_ACTIVE;
  ASSERT_TRUE(GetExitCodeProcess(process_info.hProcess, &exit_code));
  CloseHandle(process_info.hThread);
  CloseHandle(process_info.hProcess);

  EXPECT_NE(exit_code, 0U);
  ASSERT_TRUE(std::filesystem::exists(artifact));
  std::ifstream artifact_file(artifact);
  ASSERT_TRUE(artifact_file.is_open());
  std::string exception_code;
  std::getline(artifact_file, exception_code);
  EXPECT_FALSE(exception_code.empty());
  std::filesystem::remove(artifact, error);
}
#endif

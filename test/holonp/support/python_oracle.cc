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

#include "python_oracle.hh"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

namespace holonp_test {

namespace {

std::string dtype_name(holoflow::core::DType d) {
  switch (d) {
  case holoflow::core::DType::U8:
    return "U8";
  case holoflow::core::DType::U16:
    return "U16";
  case holoflow::core::DType::F32:
    return "F32";
  case holoflow::core::DType::CF32:
    return "CF32";
  }
  throw std::invalid_argument("dtype_name: unknown dtype");
}

std::filesystem::path make_oracle_tmpdir() {
  static std::atomic<int> counter{0};
  const auto              base = std::filesystem::temp_directory_path();
  const auto              now  = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto name = "holonp_oracle_" + std::to_string(now) + "_" + std::to_string(++counter);
  const auto path = base / name;
  std::filesystem::create_directories(path);
  return path;
}

} // namespace

OracleOutput invoke_oracle(const OracleInput &input, const std::filesystem::path &oracle_script) {
  const auto tmpdir = make_oracle_tmpdir();

  // Write input payloads and build manifest.
  nlohmann::json manifest;
  manifest["op"]     = input.op;
  manifest["inputs"] = nlohmann::json::array();

  for (size_t i = 0; i < input.input_descs.size(); ++i) {
    const auto &desc    = input.input_descs[i];
    const auto &bytes   = input.input_bytes[i];
    const auto  payload = "input_" + std::to_string(i) + ".bin";

    {
      std::ofstream f(tmpdir / payload, std::ios::binary);
      f.write(reinterpret_cast<const char *>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    }

    nlohmann::json entry;
    entry["shape"]   = desc.shape;
    entry["dtype"]   = dtype_name(desc.dtype);
    entry["strides"] = desc.strides;
    entry["offset"]  = desc.offset;
    entry["payload"] = payload;
    manifest["inputs"].push_back(entry);
  }

  // Op-specific settings (may be an empty object for ops that need none).
  manifest["settings"] = input.settings;

  // List expected output payload filenames.
  nlohmann::json out_payloads = nlohmann::json::array();
  for (size_t i = 0; i < input.n_outputs; ++i) {
    out_payloads.push_back("output_" + std::to_string(i) + ".bin");
  }
  manifest["output_payloads"] = out_payloads;

  // Write manifest.
  const auto manifest_path = tmpdir / "manifest.json";
  {
    std::ofstream f(manifest_path);
    f << manifest.dump(2);
  }

  // Invoke Python oracle.
  std::ostringstream cmd;
  cmd << "python \"" << oracle_script.string() << "\" \"" << manifest_path.string() << "\"";
  const int ret = std::system(cmd.str().c_str());
  if (ret != 0) {
    std::filesystem::remove_all(tmpdir);
    throw std::runtime_error("Python oracle failed (exit " + std::to_string(ret) +
                             ").\nCommand: " + cmd.str());
  }

  // Read back outputs.
  OracleOutput result;
  result.output_bytes.reserve(input.n_outputs);
  for (size_t i = 0; i < input.n_outputs; ++i) {
    const auto    out_path = tmpdir / ("output_" + std::to_string(i) + ".bin");
    std::ifstream f(out_path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
      std::filesystem::remove_all(tmpdir);
      throw std::runtime_error("Oracle output not found: " + out_path.string());
    }
    const auto sz = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<std::byte> buf(sz);
    f.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(sz));
    result.output_bytes.push_back(std::move(buf));
  }

  std::filesystem::remove_all(tmpdir);
  return result;
}

} // namespace holonp_test

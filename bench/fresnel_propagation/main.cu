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

#include "support.cuh"

int main(int argc, char **argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv))
    return 1;
  benchmark::RegisterBenchmark("Fresnel/FP32", [](benchmark::State &s) { run(&s, false); })
      ->UseManualTime()
      ->Unit(benchmark::kMicrosecond);
  benchmark::RegisterBenchmark("Fresnel/FP16_Autotuned", [](benchmark::State &s) { run(&s, true); })
      ->UseManualTime()
      ->Unit(benchmark::kMicrosecond);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return failed ? 1 : 0;
}

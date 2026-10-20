// Copyright 2025 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef HOLOFLOW_TEST_PERFETTO
#pragma push_macro("min")
#pragma push_macro("max")
#undef min
#undef max
#include <perfetto.h>
#pragma pop_macro("max")
#pragma pop_macro("min")
#endif

namespace trace_test {
struct Slice {
  std::string name;
  uint64_t    track;
  bool        complete = false;
  std::string outcome;
};
struct Trace {
  std::vector<Slice>              slices;
  std::map<uint64_t, std::string> threads;
};

inline Trace read(const std::filesystem::path &path) {
  Trace result;
#ifdef HOLOFLOW_TEST_PERFETTO
  using namespace perfetto::protos::pbzero;
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("Missing native trace: " + path.string());
  const std::string bytes{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  Trace_Decoder     trace(bytes);
  struct Sequence {
    uint64_t                        track = 0;
    std::map<uint64_t, std::string> names;
    std::map<uint64_t, std::string> annotations;
  };
  std::map<uint32_t, Sequence>            sequences;
  std::map<uint64_t, std::vector<size_t>> stacks;
  for (auto packets = trace.packet(); packets; ++packets) {
    TracePacket_Decoder packet(*packets);
    auto               &sequence = sequences[packet.trusted_packet_sequence_id()];
    if (packet.sequence_flags() & 1)
      sequence = {};
    if (packet.has_track_descriptor()) {
      TrackDescriptor_Decoder descriptor(packet.track_descriptor());
      if (descriptor.has_thread()) {
        ThreadDescriptor_Decoder thread(descriptor.thread());
        result.threads[descriptor.uuid()] = thread.thread_name().ToStdString();
      }
    }
    if (packet.has_trace_packet_defaults()) {
      TracePacketDefaults_Decoder defaults(packet.trace_packet_defaults());
      if (defaults.has_track_event_defaults()) {
        TrackEventDefaults_Decoder event_defaults(defaults.track_event_defaults());
        sequence.track = event_defaults.track_uuid();
      }
    }
    if (packet.has_interned_data()) {
      InternedData_Decoder interned(packet.interned_data());
      for (auto names = interned.event_names(); names; ++names) {
        EventName_Decoder name(*names);
        sequence.names[name.iid()] = name.name().ToStdString();
      }
      for (auto names = interned.debug_annotation_names(); names; ++names) {
        DebugAnnotationName_Decoder name(*names);
        sequence.annotations[name.iid()] = name.name().ToStdString();
      }
    }
    if (!packet.has_track_event())
      continue;
    TrackEvent_Decoder event(packet.track_event());
    const auto         track_id = event.has_track_uuid() ? event.track_uuid() : sequence.track;
    auto              &stack    = stacks[track_id];
    if (event.type() == TrackEvent::TYPE_SLICE_BEGIN) {
      const auto name =
          event.has_name() ? event.name().ToStdString() : sequence.names[event.name_iid()];
      stack.push_back(result.slices.size());
      result.slices.push_back({name, track_id});
    } else if (event.type() == TrackEvent::TYPE_SLICE_END) {
      if (stack.empty())
        throw std::runtime_error("Unmatched native slice end");
      auto &slice    = result.slices[stack.back()];
      slice.complete = true;
      stack.pop_back();
      for (auto annotations = event.debug_annotations(); annotations; ++annotations) {
        DebugAnnotation_Decoder annotation(*annotations);
        const auto name = annotation.has_name() ? annotation.name().ToStdString()
                                                : sequence.annotations[annotation.name_iid()];
        if (name == "outcome")
          slice.outcome = annotation.string_value().ToStdString();
      }
    }
  }
#else
  (void)path;
#endif
  return result;
}

inline std::filesystem::path temporary_path(std::string_view prefix) {
  return std::filesystem::temp_directory_path() /
         (std::string(prefix) +
          std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
}
} // namespace trace_test

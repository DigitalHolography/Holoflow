#include <QApplication>
#include <boost/graph/graphviz.hpp>
#include <nlohmann/json.hpp>
#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <vector>

#include "bug_buster/bug_buster.hh"
#include "holoflow/v3/model/compiler.hh"
#include "holoflow/v3/model/descriptor.hh"
#include "holoflow/v3/model/runner.hh"
#include "holovibes/accumulators/batched_spsc_accumulator.hh"
#include "holovibes/accumulators/sliding_average_accumulator.hh"
#include "holovibes/holovibes.hh"
#include "holovibes/sinks/qt_display_sink.hh"
#include "holovibes/sources/holofile_source.hh"
#include "holovibes/tasks/angular_spectrum_task.hh"
#include "holovibes/tasks/average_task.hh"
#include "holovibes/tasks/convert_task.hh"
#include "holovibes/tasks/fft_shift_task.hh"
#include "holovibes/tasks/fresnel_diffraction_task.hh"
#include "holovibes/tasks/identity_task.hh"
#include "holovibes/tasks/pca_task.hh"
#include "holovibes/tasks/percentile_clip_task.hh"
#include "holovibes/tasks/stft_task.hh"
#include "holovibes/ui/main_window.hh"
#include "holovibes/ui/tensor_display_widget.hh"

using json = nlohmann::json;

void setup_global_logger() {
  constexpr std::size_t queue_size = 8192;
  constexpr std::size_t num_threads = 1;
  spdlog::init_thread_pool(queue_size, num_threads);

  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  std::vector<spdlog::sink_ptr> sinks{console_sink};

  auto global_logger = std::make_shared<spdlog::logger>(
      "global_logger", sinks.begin(), sinks.end());

  spdlog::set_default_logger(global_logger);
  spdlog::set_level(spdlog::level::info);
  spdlog::flush_on(spdlog::level::warn);
  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [thread %t] [%n] [%^%l%$] %v");
}

int main(int argc, char **argv) {
  try {
    setup_global_logger();

    QApplication app(argc, argv);
    holovibes::ui::MainWindow main_window;
    main_window.show();
    return app.exec();

  } catch (std::exception &e) {
    dh::holovibes_logger()->critical("{}", e.what());
  }
}
#pragma once

#include <memory>
#include <spdlog/spdlog.h>

namespace dh {

std::shared_ptr<spdlog::logger> &curaii_logger();

}
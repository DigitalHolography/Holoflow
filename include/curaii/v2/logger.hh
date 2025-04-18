#pragma once

#include <memory>
#include <spdlog/spdlog.h>

namespace curaii {

std::shared_ptr<spdlog::logger> &logger();

}
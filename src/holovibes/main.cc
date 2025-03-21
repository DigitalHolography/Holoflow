#include <glog/logging.h>

#include "holoflow/model_descriptor.hh"

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = 1;
  FLAGS_v = 1;
  FLAGS_colorlogtostderr = true;
  google::InitGoogleLogging(argv[0]);

  LOG(INFO) << "Welcome to Holovibes!";
}
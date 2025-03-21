#include <glog/logging.h>

#include "holoflow/model.hh"
#include "holoflow/model_descriptor.hh"
#include "holoflow/tensor.hh"
#include "holovibes/accumulators/batched_spsc_accumulator.hh"

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = 1;
  FLAGS_v = 2;
  FLAGS_colorlogtostderr = true;
  google::InitGoogleLogging(argv[0]);

  LOG(INFO) << "Welcome to Holovibes!";

  dh::ModelDescriptor descriptor;

  // ==========================================================================
  //                     Add factories
  // ==========================================================================

  descriptor.add_accumulator_factory(
      "BatchedSPSCAccumulator",
      std::make_unique<dh::BatchedSPSCAccumulatorFactory>());

  // ==========================================================================
  //                     Add nodes
  // ==========================================================================

  descriptor.add_accumulator("BatchedSPSCAccumulator", "input_accumulator",
                             R"({
    "nb_slots": 1024,
    "dequeue_batch_size": 32
  })"_json);

  // ==========================================================================
  //                     Link nodes
  // ==========================================================================

  descriptor.set_root_accumulator("input_accumulator");

  // ==========================================================================
  //                     Build model
  // ==========================================================================

  dh::TensorMeta meta(dh::DataType::U8, dh::MemoryLocation::DEVICE,
                      {32, 340, 512});

  auto model = dh::Model::from_descriptor(descriptor, meta).value();

  // ==========================================================================
  //                     Run model
  // ==========================================================================
}
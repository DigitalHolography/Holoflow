# Copyright 2026 Digital Holography Foundation
# SPDX-License-Identifier: Apache-2.0

"""Share the host-transfer queue between demo outputs."""

import holoflow as hf


def host_queue(image, config, node_id):
    return hf.batch_queue(
        image.to("host"), config.queues["output_images"], 1, 1, node_id=node_id
    )


def propagation_queue(field, config, *, precision):
    capacity = config.queues["propagation_windows"]
    if precision == "fp16":
        current, _, _ = hf.dual_reader_batch_queue(
            field, capacity, config.postprocessing["average_window"]
        )
        return current
    return hf.batch_queue(field, capacity, 1, 1, node_id="propagation_windows")


def output_queue(image, config, *, precision):
    collect_images = config.viewer["enabled"] and not config.viewer["cao_coefficients_only"]
    if precision == "fp16" and not collect_images:
        return hf.batch_queue(image, config.queues["output_images"], 1, 1, node_id="device_output")
    return host_queue(image, config, "host_output")

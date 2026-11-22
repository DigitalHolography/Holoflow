# Copyright 2026 Digital Holography Foundation
# SPDX-License-Identifier: Apache-2.0

"""Share the host-transfer queue between demo outputs."""

import holoflow as hf


def host_queue(image, config, node_id):
    return hf.batch_queue(
        image.to("host"), config.queues["output_images"], 1, 1, node_id=node_id
    )

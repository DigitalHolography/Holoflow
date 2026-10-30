# Copyright 2026 Digital Holography Foundation
# SPDX-License-Identifier: Apache-2.0
file(READ "${INPUT}" SOURCE)
string(FIND "${SOURCE}" ")HF_KERNEL\"" COLLISION)
if(NOT COLLISION EQUAL -1)
    message(FATAL_ERROR "Embedded CUDA source contains reserved delimiter")
endif()
get_filename_component(SOURCE_NAME "${INPUT}" NAME)
file(WRITE "${OUTPUT}" "#pragma once\nnamespace holotask::detail {\ninline constexpr char ${SYMBOL}[] = R\"HF_KERNEL(#line 1 \"${SOURCE_NAME}\"\n${SOURCE})HF_KERNEL\";\n}\n")

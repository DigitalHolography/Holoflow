# Copyright 2026 Digital Holography Foundation
# SPDX-License-Identifier: Apache-2.0

"""Linear algebra operations for symbolic tensors."""

from ._graph import Tensor, operation


def eigh(a: Tensor, UPLO: str = "L", *, node_id: str | None = None):
    """Return ascending eigenvalues and column eigenvectors of real symmetric matrices."""
    return operation("Eigh", (a,), {"uplo": UPLO}, node_id)

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

__all__ = []
__version__ = "0.10.2"

import sys

# backwards compatibility to support `from fairseq.meters import AverageMeter`
from fairseq.logging import meters # noqa

sys.modules["fairseq.meters"] = meters

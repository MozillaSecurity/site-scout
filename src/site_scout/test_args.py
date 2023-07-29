# type: ignore
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
from .args import parse_args


def test_args_01():
    """test parse_args()"""
    assert parse_args(["fake_bin", "-i", "fake_input"])

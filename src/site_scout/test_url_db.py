# type: ignore
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
from pytest import mark, raises

from .url_db import UrlDBError, verify_db


@mark.parametrize(
    "data",
    [
        # valid
        {"d": {"s": ["/"]}},
        # valid
        {"1": {"2": ["/"]}},
        # valid
        {"1d": {"s2": ["/"]}},
        # valid no subdomin
        {"d": {"": ["/"]}},
    ],
)
def test_verify_db(data):
    """test verify_db()"""
    verify_db(data)


@mark.parametrize(
    "data, msg",
    [
        # not a dict
        ([], "Data must be stored in a dict"),
        # empty domain name
        ({"": {"s": ["/"]}}, "Domain must be a string"),
        # subdomain is not a string
        ({"d": {None: ["/"]}}, "Subdomain must be a string"),
        # empty domain entry
        ({"d": {}}, "Invalid domain entry: 'd'"),
        # empty subdomain entry
        ({"d": {"s": []}}, "Invalid subdomain entry: 's' in 'd'"),
        # empty path entry
        ({"d": {"s": [""]}}, "Path must be a string starting with '/'"),
        # upper case value in domain
        ({"D": {"s": ["/"]}}, "Domain must be lower case: 'D'"),
        # upper case value in subdomain
        ({"d": {"S": ["/"]}}, "Subdomain must be lower case: 'S' in 'd'"),
    ],
)
def test_verify_db_failures(data, msg):
    """test verify_db() failures"""
    with raises(UrlDBError, match=msg):
        verify_db(data)

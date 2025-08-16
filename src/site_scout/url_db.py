# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import annotations

from typing import Any, NewType, cast

# Note: Python 3.10+ use TypeAlias
UrlDB = NewType("UrlDB", dict[str, dict[str, list[str]]])


class UrlDBError(Exception):
    """Raised when UrlDB related issues are detected."""


def verify_db(data: Any) -> UrlDB:
    """Verify UrlDB structure and data.

    Args:
        data: Data to check.

    Return:
        UrlDB.
    """
    if not isinstance(data, dict):
        raise UrlDBError("Data must be stored in a dict")
    # check domains
    for domain, subdomains in data.items():
        if not isinstance(domain, str) or not domain:
            raise UrlDBError("Domain must be a string")
        if not isinstance(subdomains, dict) or not subdomains:
            raise UrlDBError(f"Invalid domain entry: '{domain}'")
        if any(x.isupper() for x in domain):
            raise UrlDBError(f"Domain must be lower case: '{domain}'")
        # check subdomains
        for subdomain, paths in subdomains.items():
            if not isinstance(subdomain, str):
                raise UrlDBError("Subdomain must be a string")
            if not isinstance(paths, list) or not paths:
                raise UrlDBError(
                    f"Invalid subdomain entry: '{subdomain}' in '{domain}'"
                )
            if any(x.isupper() for x in subdomain):
                raise UrlDBError(
                    f"Subdomain must be lower case: '{subdomain}' in '{domain}'"
                )
            # check paths
            for path in paths:
                if not isinstance(path, str) or not path.startswith("/"):
                    raise UrlDBError("Path must be a string starting with '/'")
    return cast("UrlDB", data)

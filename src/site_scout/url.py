# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import annotations

from hashlib import sha1
from logging import WARNING, getLogger
from re import compile as re_compile
from string import punctuation
from urllib.parse import quote, urlsplit

from tldextract import extract

# suppress output from filelock (used by tldextract)
getLogger("filelock").setLevel(WARNING)


class URLParseError(Exception):
    """Provided input cannot be converted to a URL."""


class URL:
    """URL components."""

    ALLOWED_SCHEMES = frozenset(("http", "https"))
    VALID_DOMAIN = re_compile(r"[a-zA-Z0-9:.-]+")
    VALID_SUBDOMAIN = re_compile(r"[a-zA-Z0-9_.-]+")

    __slots__ = ("_uid", "alias", "domain", "path", "scheme", "subdomain")

    def __init__(
        self,
        domain: str,
        subdomain: str = "",
        path: str = "/",
        scheme: str = "http",
    ) -> None:
        self.alias: str | None = None
        self.domain = domain
        self.path = path
        self.scheme = scheme
        self.subdomain = subdomain
        self._uid: str | None = None

    def __str__(self) -> str:
        if not self.subdomain:
            return f"{self.scheme}://{self.domain}{self.path}"
        return f"{self.scheme}://{self.subdomain}.{self.domain}{self.path}"

    @classmethod
    def create(
        cls,
        domain: str,
        subdomain: str = "",
        path: str = "/",
        scheme: str = "http",
    ) -> URL:
        """Sanitize, verify data and create a URL if possible.

        Args:
            domain: Domain.
            subdomain: Subdomain.
            path: Path, must begin with '/'.
            scheme: Scheme.

        Returns:
            URL object if input is valid otherwise None.
        """
        scheme = scheme.lower()
        if scheme not in cls.ALLOWED_SCHEMES:
            raise URLParseError(f"Unsupported scheme '{scheme}'")

        try:
            # use idna to encode domain with non ascii characters
            domain = domain.lower().encode("idna").decode("ascii")
        except UnicodeError:
            raise URLParseError(f"Invalid domain '{domain}'") from None
        if cls.VALID_DOMAIN.fullmatch(domain) is None:
            raise URLParseError(f"Invalid domain '{domain}'")

        if subdomain:
            try:
                subdomain = subdomain.lower().encode("idna").decode("ascii")
            except UnicodeError:
                raise URLParseError(f"Invalid subdomain '{subdomain}'") from None
            if cls.VALID_SUBDOMAIN.fullmatch(subdomain) is None:
                raise URLParseError(f"Invalid subdomain '{subdomain}'")

        if not path.startswith("/"):
            raise URLParseError("Path must begin with '/'")
        # percent encode non ascii characters in path if needed
        if not path.isascii():
            path = quote(path, safe=punctuation)

        return cls(domain, subdomain=subdomain, path=path, scheme=scheme)

    # pylint: disable=too-many-branches
    @classmethod
    def parse(cls, url: str) -> URL:
        """Parse URL from a given string. Only URLs with a valid domain and tld are
        supported. IP addresses, `about:` pages, etc, are not supported.

        Args:
            url: Input to parse.

        Returns:
            URL object if string is successfully parsed.
        """
        url = url.strip()
        if "://" not in url:
            url = f"http://{url}"
        try:
            parsed = urlsplit(url, allow_fragments=False)
        except ValueError:
            raise URLParseError("Invalid URL") from None

        # strip credentials
        if "@" in parsed.netloc:
            netloc = parsed.netloc.rsplit("@")[-1]
        else:
            netloc = parsed.netloc

        # parse domain, subdomain and tld
        udi = extract(netloc)
        if not udi.domain:
            raise URLParseError("Missing domain")
        if not udi.suffix:
            raise URLParseError("Missing suffix")
        if not udi.subdomain and netloc[0] == ".":
            raise URLParseError("Invalid subdomain")

        # parse port
        if ":" in netloc:
            try:
                port = int(netloc.rsplit(":", maxsplit=1)[-1])
            except ValueError:
                raise URLParseError("Port must be a number") from None
            if port < 1 or port > 65535:
                raise URLParseError("Invalid port number")
            domain = f"{udi.domain}.{udi.suffix}:{port}"
        else:
            domain = f"{udi.domain}.{udi.suffix}"

        # parse path
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        # sanitize parsed data and create URL
        return URL.create(
            domain,
            subdomain=udi.subdomain,
            path=path,
            scheme=parsed.scheme,
        )

    @property
    def uid(self) -> str:
        """Unique ID.

        Args:
            None

        Returns:
            Unique ID.
        """
        # this does NOT need to be cryptographically secure
        # it needs to be filesystem safe and *somewhat* unique
        if not self._uid:
            self._uid = sha1(
                str(self).encode(errors="replace"), usedforsecurity=False
            ).hexdigest()
        return self._uid

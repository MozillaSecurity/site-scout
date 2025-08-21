# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from bisect import insort
from logging import DEBUG, INFO, getLogger
from pathlib import Path
from typing import TYPE_CHECKING, cast

from yaml import dump

try:
    from yaml import CSafeDumper as SafeDumper
except ImportError:
    from yaml import SafeDumper  # type: ignore

from .main import init_logging, load_yml
from .url import URL, URLParseError
from .url_db import UrlDBError

if TYPE_CHECKING:
    from collections.abc import Generator

    from .url_db import UrlDB

LOG = getLogger(__name__)


class UrlCollection:
    """Manage collections of URLs."""

    __slots__ = ("_db", "unparsable")

    def __init__(self, url_db: UrlDB | None = None) -> None:
        self._db = url_db or cast("UrlDB", {})
        self.unparsable: set[str] = set()

    def __len__(self) -> int:
        count = 0
        for domain in self._db:
            for subdomain in self._db[domain]:
                count += len(self._db[domain][subdomain])
        return count

    def __iter__(self) -> Generator[URL]:
        for domain in self._db:
            for subdomain in self._db[domain]:
                for path in self._db[domain][subdomain]:
                    yield URL(domain, subdomain=subdomain, path=path)

    def add_list(self, urls_file: Path) -> int:
        """Load a text file containing a line separated list of URLs and add them
        to the collection of URLs.

        Args:
            urls_file: File to load.

        Returns:
            Number of previously unknown URLs that have been added.
        """
        added = 0
        with urls_file.open() as in_fp:
            for line in in_fp:
                entry = line.strip()
                if not entry or entry[0] == "#":
                    # skip comments and empty lines
                    continue
                if self.add_str(entry) is not None:
                    added += 1
        return added

    def add_str(self, url: str) -> URL | None:
        """Parse URL from a given string and add it to the collection of known URLs.

        Args:
            url: URL to add.

        Returns:
            URL that was added.
        """
        try:
            parsed = URL.parse(url)
        except URLParseError:
            LOG.debug("failed to parse and add: '%s'", url)
            self.unparsable.add(url)
            return None
        return parsed if self.add_url(parsed) else None

    def add_url(self, url: URL) -> bool:
        """Add a URL to the collection of known URLs.

        Args:
            url: URL to add.

        Returns:
            True if the URL was added otherwise False.
        """
        if url.domain not in self._db:
            self._db[url.domain] = {}
        if url.subdomain not in self._db[url.domain]:
            self._db[url.domain][url.subdomain] = []
        if url.path not in self._db[url.domain][url.subdomain]:
            insort(self._db[url.domain][url.subdomain], url.path)
            LOG.debug("added: %s", url)
            return True
        return False

    @property
    def domains(self) -> Generator[str]:
        """All known domains.

        Args:
            None

        Yields:
            Domains in URL collection.
        """
        yield from self._db

    def count_entries(self) -> dict[str, int]:
        """Count entries for each domain.

        Args:
            None

        Returns:
            A mapping of domains and entry counts.
        """
        results: dict[str, int] = {}
        for domain in self._db:
            count = 0
            for sub in self._db[domain]:
                count += len(self._db[domain][sub])
            results[domain] = count
        return results

    def remove_url(self, url: str) -> bool:
        """Remove a URL from the collection.

        Args:
            url: URL to remove.

        Returns:
            True if the URL was removed otherwise False.
        """
        try:
            parsed = URL.parse(url)
        except URLParseError:
            LOG.debug("failed to parse and remove: '%s'", url)
            return False

        try:
            self._db[parsed.domain][parsed.subdomain].remove(parsed.path)
        except (KeyError, ValueError):
            LOG.debug("failed to remove unknown url: '%s'", url)
            return False

        # remove empty domain/subdomain entries
        if not self._db[parsed.domain][parsed.subdomain]:
            del self._db[parsed.domain][parsed.subdomain]
        if not self._db[parsed.domain]:
            del self._db[parsed.domain]

        return True

    @classmethod
    def load_yml(cls, src: Path) -> UrlCollection | None:
        """Load UrlCollection from a YML file.

        Args:
            src: File to load.

        Returns:
            UrlCollection.
        """
        try:
            return cls(load_yml(src))
        except UrlDBError as exc:
            LOG.warning("Load failure - %s (ignored: %s)", exc, src.resolve())
        return None

    def save_yml(self, dst: Path) -> None:
        """Save UrlCollection to a YML file.

        Args:
            dst: File to save to.

        Returns:
            None
        """
        with dst.open("w") as out_fp:
            dump(self._db, out_fp, Dumper=SafeDumper, default_style="'")

    def sort_paths(self) -> None:
        """Sort paths.

        Args:
            None

        Returns:
            None.
        """
        for subdomains in self._db.values():
            for paths in subdomains.values():
                paths.sort()


def parse_args(argv: list[str] | None = None) -> Namespace:
    """Argument parsing"""
    parser = ArgumentParser(description="Manage URL collections used by Site-scout.")
    parser.add_argument("url_db", type=Path, help="YML file containing URL data.")
    parser.add_argument(
        "--disable-logging",
        action="store_true",
        help="Disable console output before running (default: %(default)s).",
    )
    parser.add_argument("-d", "--display-list", action="store_true")
    parser.add_argument(
        "--domain-entries",
        type=int,
        default=0,
        help="Display domains with the most entries (default: %(default)s).",
    )
    parser.add_argument(
        "--force-save",
        action="store_true",
        help="Save data even if no changes have been made (refresh YML file).",
    )
    level_map = {"INFO": INFO, "DEBUG": DEBUG}
    parser.add_argument(
        "--log-level",
        choices=sorted(level_map),
        default="INFO",
        help="Configure console logging (default: %(default)s).",
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "-l",
        "--url-list",
        type=Path,
        help="A .yml file URL database or a .txt file containing line separated list"
        "of URLs.",
    )
    group.add_argument("-r", "--remove", help="Remove single URL.")
    group.add_argument("-u", "--url", help="Add single URL.")
    args = parser.parse_args(argv)

    args.log_level = level_map[args.log_level]

    if args.url_list and not args.url_list.is_file():
        parser.error(f"{args.url_list} is not a file.")

    return args


# pylint: disable=too-many-branches
def main(argv: list[str] | None = None) -> int:
    """Main function"""
    args = parse_args(argv)
    init_logging(args.log_level, args.disable_logging)

    # pylint: disable=too-many-nested-blocks
    try:
        # load data
        if args.url_db and args.url_db.is_file():
            LOG.info("Loading '%s'...", args.url_db)
            urls = UrlCollection.load_yml(args.url_db)
            if urls is None:
                return 1
            urls.sort_paths()
        else:
            urls = UrlCollection()

        # update data
        changed = False
        if args.url:
            added = urls.add_str(args.url)
            if added is not None:
                LOG.info("Added '%s'", added)
                changed = True
        elif args.url_list:
            LOG.info("Scanning URL(s) in '%s'...", args.url_list)
            if args.url_list.suffix.lower() == ".yml":
                # attempt to load URLs from a YML file
                merge_urls = UrlCollection.load_yml(args.url_list)
                if merge_urls is not None:
                    added_count = 0
                    for entry in merge_urls:
                        if urls.add_url(entry):
                            added_count += 1
            else:
                # attempt to load a line separated list of URLs
                added_count = urls.add_list(args.url_list)
            LOG.info("Added %s URL(s).", f"{added_count:,d}")
            changed = added_count > 0
        elif args.remove:
            changed = urls.remove_url(args.remove)
            if changed:
                LOG.info("Removed 1 URL.")

        # save data
        if changed or args.force_save:
            LOG.info("Saving '%s'...", args.url_db)
            urls.save_yml(args.url_db)
        else:
            LOG.info("No change made.")

        # show current list of URLs
        if args.display_list:
            LOG.info("Known URLs:")
            for entry in urls:
                LOG.info("> %s", entry)

        # show unparsable input
        if urls.unparsable:
            LOG.warning("Unparsable entries: %s", f"{len(urls.unparsable):,d}")
            for unparsable in urls.unparsable:
                LOG.info("> '%s'", unparsable)

        # largest domains
        if args.domain_entries > 0:
            LOG.info("Domains with most entries:")
            for domain, count in sorted(
                urls.count_entries().items(), key=lambda kv: kv[1], reverse=True
            )[: args.domain_entries]:
                LOG.info("> %d '%s'", count, domain)

        # show summary
        try:
            file_size = args.url_db.stat().st_size
        except FileNotFoundError:
            file_size = 0
        LOG.info(
            "%s URLs (%s domains) in db (%s bytes).",
            f"{len(urls):,d}",
            f"{sum(1 for _ in urls.domains):,d}",
            f"{file_size:,d}",
        )

    except KeyboardInterrupt:  # pragma: no cover
        LOG.warning("Aborting...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from logging import DEBUG, INFO, getLogger
from pathlib import Path
from typing import cast

from .main import init_logging
from .site_scout import NO_SUBDOMAIN, UrlDB
from .url_collection import UrlCollection

LOG = getLogger(__name__)

DOMAIN_OVERHEAD = 4
PATH_OVERHEAD = 7
SUBDOMAIN_OVERHEAD = 6
TARGET_SIZE = 350_000


def split_collection(
    urls: UrlCollection, dst: Path, prefix: str, size: int = TARGET_SIZE
) -> int:
    """Split a URL collection and write it to multiple files.

    Args:
        urls: URL collection to split.
        dst: Output directory.
        prefix: Filename prefix.
        size: Target file size of an output file.

    Returns:
        Number of files created.
    """
    # urls to add to file
    current_batch = cast(UrlDB, {})
    # used to calculate projected file size
    current_size = 0
    file_num = 0
    for url in urls:
        # add url to batch
        if url.domain not in current_batch:
            current_size += len(url.domain) + DOMAIN_OVERHEAD
            current_batch[url.domain] = {}
        subdomain = url.subdomain or NO_SUBDOMAIN
        if subdomain not in current_batch[url.domain]:
            current_size += len(subdomain) + SUBDOMAIN_OVERHEAD
            current_batch[url.domain][subdomain] = []
        current_size += len(url.path) + PATH_OVERHEAD
        current_batch[url.domain][subdomain].append(url.path)
        # output current batch of urls to a file
        if current_size >= size:
            file_num += 1
            UrlCollection(current_batch).save_yml(dst / f"{prefix}{file_num:03d}.yml")
            current_size = 0
            current_batch.clear()
    # output any remaining urls to a file
    if current_batch:
        file_num += 1
        UrlCollection(current_batch).save_yml(dst / f"{prefix}{file_num:03d}.yml")
    return file_num


def parse_args(argv: list[str] | None = None) -> Namespace:
    """Argument parsing"""
    parser = ArgumentParser(description="Split a URL collection used by Site-scout.")
    parser.add_argument("url_db", type=Path, help="YML file containing data.")
    parser.add_argument(
        "-o", "--output", default=Path.cwd(), type=Path, help="Output directory."
    )
    parser.add_argument(
        "-p",
        "--prefix",
        help="Output file prefix. Using 'out-' will yield 'out-01.yml'...",
    )
    parser.add_argument(
        "-s",
        "--size",
        default=TARGET_SIZE,
        type=int,
        help="Target output size for parts in bytes. (default: %(default)s).",
    )
    level_map = {"INFO": INFO, "DEBUG": DEBUG}
    parser.add_argument(
        "--log-level",
        choices=sorted(level_map),
        default="INFO",
        help="Configure console logging (default: %(default)s).",
    )
    args = parser.parse_args(argv)
    args.log_level = level_map[args.log_level]
    # check args
    if not args.url_db.is_file():
        parser.error(f"{args.url_db} is not a file.")
    if not args.output.is_dir():
        parser.error(f"{args.output} is not a directory.")
    return args


# pylint: disable=too-many-branches
def main(argv: list[str] | None = None) -> int:
    """Main function"""
    args = parse_args(argv)
    init_logging(args.log_level)

    try:
        # load data
        LOG.info("Loading '%s'...", args.url_db)
        urls = UrlCollection.load_yml(args.url_db)
        if urls is None or not any(urls):
            LOG.info("No urls loaded.")
            return 1
        prefix = args.prefix or f"{args.url_db.stem}-"
        LOG.info("Writing output...")
        files_created = split_collection(urls, args.output, prefix, size=args.size)
        LOG.info("%d file(s) created.", files_created)

    except KeyboardInterrupt:  # pragma: no cover
        LOG.warning("Aborting...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

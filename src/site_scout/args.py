# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from argparse import ArgumentParser, Namespace
from importlib.metadata import PackageNotFoundError, version
from logging import DEBUG, INFO
from os import getenv
from pathlib import Path
from platform import system
from typing import List, Optional

try:
    __version__ = version("site-scout")
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = "unknown"


def is_headless() -> bool:
    """Detect if running in a headless environment.

    Args:
        None

    Returns:
        True if running in a headless environment otherwise False.
    """
    return (
        system() == "Linux" and not getenv("DISPLAY") and not getenv("WAYLAND_DISPLAY")
    )


def parse_args(argv: Optional[List[str]] = None) -> Namespace:
    """Argument parsing"""
    parser = ArgumentParser(
        prog="site-scout", description="Visit provided URLs and report results."
    )
    parser.add_argument("binary", type=Path, help="Firefox binary to use.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-i",
        "--input",
        default=[],
        nargs="+",
        type=Path,
        help="File(s) containing URLs to visit.",
    )
    group.add_argument("-u", "--url", default=[], nargs="+", help="URL(s) to visit.")

    display_choices = ["headless"]
    if system() == "Linux":
        display_choices.append("xvfb")
    parser.add_argument(
        "--display",
        choices=display_choices,
        default="xvfb" if is_headless() else None,
        help="Display mode.",
    )
    parser.add_argument(
        "--fuzzmanager", action="store_true", help="Report results to FuzzManager."
    )
    parser.add_argument(
        "--jobs", type=int, default=1, help="Maximum number of browsers to run at once."
    )
    parser.add_argument(
        "--launch-timeout",
        type=int,
        default=300,
        help="Time in seconds to wait for browser to launch and begin navigation"
        " (default: %(default)s).",
    )
    level_map = {"INFO": INFO, "DEBUG": DEBUG}
    parser.add_argument(
        "--log-level",
        choices=sorted(level_map),
        default="INFO",
        help="Configure console logging (default: %(default)s).",
    )
    parser.add_argument(
        "--log-limit",
        type=int,
        default=0,
        help="Browser log file size limit in MBs (default: no limit).",
    )
    parser.add_argument(
        "--memory-limit",
        type=int,
        default=0,
        help="Browser memory limit in MBs (default: no limit).",
    )
    parser.add_argument(
        "--prefs",
        type=Path,
        help="Custom prefs.js file to use (default: generated).",
    )
    parser.add_argument("--profile", type=Path, help="")
    parser.add_argument(
        "-o",
        "--output-path",
        default=Path.cwd(),
        type=Path,
        help="Location to save results (default: %(default)s).",
    )
    parser.add_argument(
        "--status-report",
        type=Path,
        help="Location to save periodic status report (default: no report).",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=120,
        help="Page load time limit in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--url-limit",
        type=int,
        default=0,
        help="Maximum number of URLs to visit (default: no limit).",
    )
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show version number.",
    )

    args = parser.parse_args(argv)

    if args.jobs < 1:
        parser.error("--jobs must be >= 1")

    args.log_level = level_map[args.log_level]

    if args.log_limit < 0:
        parser.error("--log-limit must be >= 0")
    args.log_limit *= 1_048_576

    if args.memory_limit < 0:
        parser.error("--memory-limit must be >= 0")
    args.memory_limit *= 1_048_576

    if not args.binary.is_file():
        parser.error(f"binary does not exist: '{args.binary}'")

    for in_file in args.input:
        if not in_file.is_file():
            parser.error(f"-i/--input does not exist: '{in_file}'")

    if not args.output_path.is_dir():
        parser.error(f"-o/--output-path does not exist: '{args.output_path}'")

    if args.prefs and not args.prefs.is_file():
        parser.error(f"--prefs does not exist: '{args.prefs}'")

    return args

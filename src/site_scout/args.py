# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import annotations

from argparse import ArgumentParser, Namespace
from importlib.metadata import PackageNotFoundError, version
from logging import DEBUG, INFO
from os import getenv
from pathlib import Path
from platform import system
from shutil import which

from ffpuppet import Debugger

try:
    __version__ = version("site-scout")
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = "unknown"

TIME_LIMIT_DEBUG = 300
TIME_LIMIT_DEFAULT = 90
TIME_LIMIT_EXPLORE = 180


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


# pylint: disable=too-many-branches,too-many-statements
def parse_args(argv: list[str] | None = None) -> Namespace:
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

    display_choices = ["default", "headless"]
    if system() == "Linux":
        display_choices.append("xvfb")
    parser.add_argument(
        "--display",
        choices=display_choices,
        default="xvfb" if is_headless() else "default",
        help="Display mode.",
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help="Use PageExplorer to interact with content.",
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
        help="Configure console output (default: %(default)s).",
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
        "--result-limit",
        type=int,
        default=0,
        help="Maximum number of results."
        " Remaining visits will be skipped if limit is hit (default: no limit).",
    )
    parser.add_argument(
        "--runtime-limit",
        type=int,
        default=0,
        help="Maximum runtime in seconds."
        " Remaining visits will be skipped if limit is hit (default: no limit).",
    )
    parser.add_argument(
        "--status-report",
        type=Path,
        help="Location to save periodic status report (default: no report).",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        help=f"Page load time limit in seconds (default: {TIME_LIMIT_DEFAULT}).",
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
    parser.add_argument(
        "--visits",
        type=int,
        default=1,
        help="Number of times to visit each URL (default: %(default)s).",
    )

    parser.set_defaults(coverage=False, debugger=Debugger.NONE)
    if system() == "Linux":
        parser.add_argument(
            "--coverage",
            action="store_true",
            help="Dump coverage data to disk. This requires a supported browser build.",
        )
        dbg_group = parser.add_mutually_exclusive_group()
        dbg_group.add_argument(
            "--pernosco",
            action="store_const",
            const=Debugger.PERNOSCO,
            dest="debugger",
            help="Use rr. Trace intended to be used with Pernosco.",
        )
        dbg_group.add_argument(
            "--rr",
            action="store_const",
            const=Debugger.RR,
            dest="debugger",
            help="Use rr.",
        )

    args = parser.parse_args(argv)

    if args.time_limit is None:
        if args.debugger != Debugger.NONE:
            args.time_limit = TIME_LIMIT_DEBUG
        elif args.explore:
            args.time_limit = TIME_LIMIT_EXPLORE
        else:
            args.time_limit = TIME_LIMIT_DEFAULT
    if args.time_limit < 1:
        parser.error("--time-limit must be > 0 (recommended minimum: 30)")

    if args.coverage:
        # GCOV_PREFIX_STRIP and GCOV_PREFIX are specific to Firefox coverage builds
        if not getenv("GCOV_PREFIX_STRIP"):
            parser.error("GCOV_PREFIX_STRIP must be set to use --coverage")
        if not getenv("GCOV_PREFIX"):
            parser.error("GCOV_PREFIX must be set to use --coverage")
        if args.jobs > 1:
            parser.error("Parallel jobs not supported with --coverage")

    if args.debugger in (Debugger.PERNOSCO, Debugger.RR):
        if args.fuzzmanager:
            parser.error("rr not supported with --fuzzmanager")
        # rr is only supported on Linux
        if not which("rr"):
            parser.error("rr is not installed")
        settings = "/proc/sys/kernel/perf_event_paranoid"
        value = int(Path(settings).read_bytes())
        if value > 1:
            parser.error(f"rr needs {settings} <= 1, but it is {value}")

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

    for in_path in args.input:
        if not in_path.exists():
            parser.error(f"-i/--input does not exist: '{in_path}'")

    if not args.output_path.is_dir():
        parser.error(f"-o/--output-path does not exist: '{args.output_path}'")

    if args.prefs and not args.prefs.is_file():
        parser.error(f"--prefs does not exist: '{args.prefs}'")

    return args

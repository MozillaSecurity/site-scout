# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from logging import DEBUG, basicConfig, getLogger
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Iterator, List, Optional

from prefpicker import PrefPicker
from yaml import safe_load
from yaml.parser import ParserError
from yaml.scanner import ScannerError

from .args import parse_args
from .site_scout import SiteScout, verify_dict

LOG = getLogger(__name__)


def init_logging(level: int) -> None:
    """Initialize logging

    Arguments:
        level: logging verbosity level

    Returns:
        None
    """
    if level == DEBUG:
        date_fmt = "%m-%d %H:%M:%S"
        log_fmt = "%(asctime)s.%(msecs)03d %(levelname).1s %(name)s | %(message)s"
    else:
        date_fmt = "%H:%M:%S"
        log_fmt = "[%(asctime)s] %(message)s"
    basicConfig(format=log_fmt, datefmt=date_fmt, level=level)


def generate_prefs(dst: Optional[Path] = None, variant: str = "a11y") -> Path:
    """Generate prefs.js file via PrefPicker.

    Arguments:
        variant: Variant to use.

    Returns:
        pref.js file.
    """
    with NamedTemporaryFile(prefix="scout-prefs-", suffix=".js", dir=dst) as tmp_fp:
        prefs = Path(tmp_fp.name)
    picker = PrefPicker.load_template(
        Path(__file__).parent / "resources" / "prefpicker.yml"
    )
    picker.create_prefsjs(prefs, variant=variant)
    return prefs


def load_input(src: List[Path]) -> Iterator[Dict[str, Dict[str, List[str]]]]:
    """Load data from filesystem.

    Arguments:
        src: Files and directories to load data from.

    Yields:
        URL data.
    """
    for entry in scan_input(src):
        LOG.debug("loading '%s'", entry)
        with entry.open("r") as in_fp:
            try:
                data = safe_load(in_fp)
            except (ParserError, ScannerError):
                LOG.warning("Load failure - Invalid yml (ignored: %s)", entry)
                continue
        err_msg = verify_dict(data)
        if err_msg:
            LOG.warning("Load failure - %s (ignored: %s)", err_msg, entry)
            continue
        yield data


def scan_input(src: List[Path]) -> Iterator[Path]:
    """Scan list of inputs which can include files or directories.

    Arguments:
        src: Paths the evaluate.

    Yields:
        URL files to load.
    """
    for entry in src:
        if entry.resolve().is_dir():
            yield from entry.glob("*.yml")
        else:
            yield entry


def main(argv: Optional[List[str]] = None) -> None:
    """Main function"""
    args = parse_args(argv)
    assert any(args.input) != any(args.url)
    init_logging(args.log_level)

    tmp_prefs = False
    try:
        # generate prefs.js file if one is not provided
        if args.prefs is None:
            args.prefs = generate_prefs()
            tmp_prefs = True

        LOG.info("Starting Site Scout...")
        with SiteScout(
            args.binary,
            profile=args.profile,
            prefs_js=args.prefs,
            debugger=args.debugger,
            display="default" if args.display == "headless" else args.display,
            launch_timeout=args.launch_timeout,
            log_limit=args.log_limit,
            memory_limit=args.memory_limit,
            extension=None,
            cert_files=None,
            fuzzmanager=args.fuzzmanager,
        ) as scout:
            for data in load_input(args.input):
                scout.load_dict(data)
            for in_url in args.url:
                scout.load_str(in_url)
            # don't randomize urls passed on the command line
            scout.schedule_urls(url_limit=args.url_limit, randomize=any(args.input))
            scout.run(
                args.output_path,
                args.time_limit,
                instance_limit=args.jobs,
                status_report=args.status_report,
            )
    except KeyboardInterrupt:
        LOG.warning("Aborting...")

    finally:
        if tmp_prefs:
            args.prefs.unlink()
        LOG.info("Done.")

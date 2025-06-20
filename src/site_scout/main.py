# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import annotations

from json import JSONDecodeError, loads
from logging import DEBUG, basicConfig, disable, getLogger
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

from ffpuppet import LaunchError
from prefpicker import PrefPicker
from yaml import load

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:  # pragma: no cover
    from yaml import SafeLoader  # type: ignore

from yaml.parser import ParserError
from yaml.reader import ReaderError
from yaml.scanner import ScannerError

from .args import parse_args
from .site_scout import SiteScout, UrlDB, verify_dict

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

LOG = getLogger(__name__)


def init_logging(level: int, disable_logging: bool) -> None:
    """Initialize logging.

    Arguments:
        level: logging verbosity level.
        disable_logging: Disable log output.

    Returns:
        None
    """
    if level == DEBUG:
        log_fmt = (
            "%(asctime)s.%(msecs)03d %(levelname).1s %(threadName)s %(name)s | "
            "%(message)s"
        )
    else:
        log_fmt = "[%(asctime)s] %(message)s"
    basicConfig(format=log_fmt, datefmt="%H:%M:%S", level=level)
    if disable_logging:
        LOG.warning("Running with logging disabled...")
        disable()


def generate_prefs(dst: Path | None = None, variant: str = "a11y") -> Path:
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


def load_jsonl(src: Path) -> Generator[tuple[str, str | None]]:
    """Load data from a jsonl file.

    The expected format is:
        {"url1": "alias1"}\n
        {"url2": "alias2"}\n
        {"url3": null}\n
        ...

    Arguments:
        src: File to load data from.

    Yields:
        URL and alias.
    """
    LOG.debug("loading (jsonl) '%s'", src.resolve())
    with src.open("r") as in_fp:
        line_no = 0
        try:
            for line_no, line in enumerate(in_fp, start=1):
                line = line.strip()
                # ignore blank lines
                if not line:
                    continue
                result = loads(line)
                # there should be exactly one entry
                if not isinstance(result, dict) or len(result) != 1:
                    LOG.error("Invalid format: '%s:%d'", src.resolve(), line_no)
                    break
                url, alias = result.popitem()
                if alias is not None and not isinstance(alias, str):
                    LOG.error("Invalid alias: '%s:%d'", src.resolve(), line_no)
                    break
                yield url, alias
        except JSONDecodeError as exc:
            LOG.error("Invalid data: '%s:%d' (%s)", src.resolve(), line_no, exc)


def load_yml(src: Path, allow_empty: bool = False) -> Generator[UrlDB]:
    """Load data from a yml file.

    Arguments:
        src: File to load data from.
        allow_empty: Empty data set is valid if True.

    Yields:
        URL data.
    """
    LOG.debug("loading (yml) '%s'", src.resolve())
    with src.open("r") as in_fp:
        try:
            data = load(in_fp, Loader=SafeLoader)
        except (ParserError, ReaderError, ScannerError):
            LOG.warning("Load failure - Invalid yml (ignored: %s)", src.resolve())
            return
    err_msg = verify_dict(data, allow_empty=allow_empty)
    if err_msg:
        LOG.warning("Load failure - %s (ignored: %s)", err_msg, src.resolve())
    else:
        yield data


def scan_input(src: Iterable[Path], suffix: str) -> Generator[Path]:
    """Scan provided locations for input files.

    Arguments:
        src: Paths to scan.
        suffix: Only files with this suffix will be included.

    Yields:
        Data files to load.
    """
    assert suffix.startswith(".")
    for entry in src:
        if entry.is_dir():
            yield from entry.glob(f"*{suffix}")
        elif entry.suffix == suffix:
            yield entry


def main(argv: list[str] | None = None) -> int:
    """Main function"""
    args = parse_args(argv)
    assert any(args.input) != any(args.url)
    init_logging(args.log_level, args.disable_logging)

    tmp_prefs = False
    try:
        # generate prefs.js file if one is not provided
        if args.prefs is None:
            args.prefs = generate_prefs()
            tmp_prefs = True

        LOG.info("Starting Site Scout...")
        with SiteScout(
            args.binary.resolve(),
            profile=args.profile,
            prefs_js=args.prefs,
            debugger=args.debugger,
            display_mode=args.display,
            launch_timeout=args.launch_timeout,
            log_limit=args.log_limit,
            memory_limit=args.memory_limit,
            extension=None,
            cert_files=None,
            fuzzmanager=args.fuzzmanager,
            coverage=args.coverage,
            explore=args.explore,
            omit_urls=args.omit_urls,
        ) as scout:
            LOG.info("Loading URLs...")
            # load jsonl files
            for src in scan_input(args.input, ".jsonl"):
                for in_url, alias in load_jsonl(src):
                    scout.load_str(in_url, alias=alias)
            # load yml files
            for src in scan_input(args.input, ".yml"):
                for data in load_yml(src):
                    scout.load_dict(data)
            # load urls from command line
            for in_url in args.url:
                scout.load_str(in_url)
            # don't randomize urls passed on the command line
            scout.schedule_urls(
                url_limit=args.url_limit,
                randomize=any(args.input),
                visits=args.visits,
            )
            scout.run(
                args.output_path,
                args.time_limit,
                instance_limit=args.jobs,
                status_report=args.status_report,
                result_limit=args.result_limit,
                runtime_limit=args.runtime_limit,
            )
    except KeyboardInterrupt:
        LOG.warning("Aborting...")

    except LaunchError:
        return 1

    finally:
        if tmp_prefs:
            args.prefs.unlink()
        LOG.info("Done.")

    return 0

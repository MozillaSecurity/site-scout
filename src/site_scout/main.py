# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from logging import DEBUG, basicConfig, getLogger
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional

from ffpuppet import Debugger
from prefpicker import PrefPicker
from yaml import safe_load

from .args import parse_args
from .site_scout import SiteScout

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


def main(argv: Optional[List[str]] = None) -> None:
    """Main function"""
    args = parse_args(argv)
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
            debugger=Debugger.NONE,
            display="default" if args.display == "headless" else args.display,
            launch_timeout=args.launch_timeout,
            log_limit=args.log_limit,
            memory_limit=args.memory_limit,
            extension=None,
            cert_files=None,
            fuzzmanager=args.fuzzmanager,
        ) as scout:
            if args.input:
                LOG.debug("loading '%s'", args.input)
                with args.input.open("r") as in_fp:
                    scout.load_dict(safe_load(in_fp))
            elif args.url:
                scout.load_str(args.url)
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

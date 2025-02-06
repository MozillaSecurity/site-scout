# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import annotations

from logging import getLogger
from pathlib import Path
from platform import machine, system
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from Collector.Collector import Collector
from FTB.ProgramConfiguration import ProgramConfiguration
from FTB.Signatures.CrashInfo import CrashInfo

LOG = getLogger(__name__)


class FuzzManagerReporter:
    """Report results to a FuzzManager server."""

    FM_CONFIG = Path.home() / ".fuzzmanagerconf"

    __slots__ = ("_conf", "_working_path")

    def __init__(
        self,
        binary: Path,
        fm_config: Path = FM_CONFIG,
        working_path: Path | None = None,
    ) -> None:
        self._working_path = working_path

        if fm_config and not fm_config.is_file():
            raise FileNotFoundError(f"Missing: {fm_config}")

        # create ProgramConfiguration that can be reported to a FM server
        if Path(f"{binary}.fuzzmanagerconf").is_file():
            # attempt to use "<binary>.fuzzmanagerconf"
            self._conf = ProgramConfiguration.fromBinary(binary)
        else:
            LOG.debug(
                "missing '%s.fuzzmanagerconf', creating ProgramConfiguration", binary
            )
            cpu = machine().lower()
            self._conf = ProgramConfiguration(
                binary.name,
                "x86_64" if cpu == "amd64" else cpu,
                system(),
            )

    @staticmethod
    def _read_ffpuppet_log(log_path: Path, log_id: str) -> list[str] | None:
        """Read logs created by FFPuppet.

        Args:
            log_path: Directory containing logs output by FFPuppet.
            log_id: Log to find. Must be: 'aux', 'stderr', 'stdout'.

        Returns:
            Lines from the requested log file.
        """
        assert log_id in ("aux", "stderr", "stdout")

        log_file: Path | None = None
        if log_id == "aux":
            # look for sanitizer logs
            for entry in log_path.glob("log_*.txt"):
                if "_asan_" in entry.name:
                    log_file = entry
                    break
            # look for minidump logs
            if not log_file:
                md_logs = sorted(log_path.glob("log_minidump_*.txt"))
                if md_logs:
                    log_file = md_logs[0]
            # TODO: valgrind logs
        else:
            log_file = log_path / f"log_{log_id}.txt"

        if log_file and log_file.is_file():
            # read text, sanitize and splitlines
            return (
                log_file.read_text("utf-8", errors="replace")
                .replace("\0", "?")
                .splitlines()
            )
        return None

    def submit(self, result: Path, metadata: dict[str, str]) -> tuple[int, str]:
        """Submit results to a FuzzManager server.

        Args:
            results: Directory containing logs to submit.
            metadata: Extra data to include in the report.

        Returns:
            Crash ID and the short signature.
        """
        collector = Collector(tool="site-scout")

        # TODO: add cache checks and report limits?

        # read the log files and create a CrashInfo object
        crash_info = CrashInfo.fromRawCrashData(
            self._read_ffpuppet_log(result, "stdout"),
            self._read_ffpuppet_log(result, "stderr"),
            self._conf,
            auxCrashData=self._read_ffpuppet_log(result, "aux"),
        )
        if metadata:
            crash_info.configuration.addMetadata(metadata)

        with TemporaryDirectory(prefix="fm-report", dir=self._working_path) as tmp_dir:
            # add result to a zip file
            zip_name = Path(tmp_dir) / f"{result.name}.zip"
            with ZipFile(zip_name, mode="w", compression=ZIP_DEFLATED) as zip_fp:
                # add test files
                for entry in result.rglob("*"):
                    if entry.is_file():
                        zip_fp.write(entry, arcname=entry.relative_to(result))

            # submit results to the FuzzManager server
            new_entry = collector.submit(
                crash_info, testCase=zip_name, testCaseQuality=5
            )

        return (int(new_entry["id"]), str(crash_info.createShortSignature()))

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import annotations

from logging import getLogger
from pathlib import Path
from shutil import copyfile, rmtree
from tempfile import mkdtemp
from time import strftime
from typing import TYPE_CHECKING

from ffpuppet import (
    BrowserTimeoutError,
    Debugger,
    DisplayMode,
    FFPuppet,
    LaunchError,
    Reason,
)

from .browser_wrapper import (
    BrowserArgs,
    BrowserState,
    BrowserWrapper,
)
from .explorer import Explorer, ExplorerMode

if TYPE_CHECKING:
    from .url import URL


LOG = getLogger(__name__)


class FirefoxWrapper(BrowserWrapper):
    """FirefoxWrapper is used to add support for Firefox."""

    __slots__ = ("_debugger", "_ffp", "_working_path")

    def __init__(self, args: BrowserArgs, working_path: Path, env_mgr: None) -> None:
        super().__init__(args, working_path, env_mgr)
        self._working_path = Path(mkdtemp(dir=working_path))
        # TODO: generate prefs if needed
        self._debugger = args.debugger or Debugger.NONE.name
        display_mode = args.display_mode or DisplayMode.DEFAULT.name
        self._ffp = FFPuppet(
            debugger=Debugger[self._debugger.upper()],
            display_mode=DisplayMode[display_mode.upper()],
            working_path=str(self._working_path),
        )

    def cleanup(self) -> None:
        self._ffp.clean_up()
        rmtree(self._working_path, ignore_errors=True)

    def close(self) -> None:
        self._ffp.close()

    def create_explorer(
        self, url: URL, load_wait: int, mode: ExplorerMode, pause: int = 0
    ) -> Explorer:
        assert self._ffp.marionette is not None
        # this can raise RuntimeError
        return Explorer(
            self.args.binary,
            self._ffp.marionette,
            str(url),
            mode=mode,
            load_wait=load_wait,
            pause=pause,
        )

    def debugger(self) -> bool:
        return self._debugger.upper() != Debugger.NONE.name

    def dump_coverage(self, timeout: int = 15) -> None:
        if self._ffp.is_healthy():
            self._ffp.dump_coverage(timeout=timeout)

    @staticmethod
    def environment_manager(instance_limit: int, browser_args: BrowserArgs) -> None:
        return None

    def is_healthy(self) -> bool:
        return self._ffp.is_healthy()

    def is_idle(self, idle_limit: int) -> bool:
        if self._ffp.is_healthy():
            # check that no processes are over the limit
            return not any(pct > idle_limit for _, pct in self._ffp.cpu_usage())
        # not running
        return True

    def launch(
        self,
        url: str,
        explore: bool,
        log_path: Path | None,
        raise_on_failure: bool,
    ) -> bool:
        success = False
        try:
            self._ffp.launch(
                self.args.binary,
                env_mod={"MOZ_CRASHREPORTER_SHUTDOWN": "1"},
                location=None if explore else str(url),
                launch_timeout=self.args.launch_timeout,
                log_limit=2 * 1024 * 1024,
                marionette=0 if explore else None,
                memory_limit=self.args.memory_limit,
                prefs_js=self.args.prefs_file,
            )
            success = True
        except LaunchError as exc:
            is_failure = not isinstance(exc, BrowserTimeoutError)
            LOG.warning("Browser launch %s...", "failure" if is_failure else "timeout")
            if raise_on_failure:
                # save failure
                if is_failure and log_path is not None:
                    self._ffp.close()
                    dst = log_path / strftime("%Y%m%d-%H%M%S-launch-failure")
                    self.save_report(dst)
                    LOG.warning("Logs saved '%s'", dst)
                raise
        finally:
            # cleanup if launch was unsuccessful
            if not success:
                self._ffp.clean_up()
        return success

    def save_report(self, dst: Path) -> None:
        self._ffp.save_logs(dst)
        # save prefs file
        if self.args.prefs_file is not None:
            copyfile(self.args.prefs_file, dst / "prefs.js")

    def state(self) -> BrowserState:
        assert self._ffp.reason is not None
        if self._ffp.reason == Reason.EXITED:
            return BrowserState.EXITED
        if self._ffp.reason in {Reason.ALERT, Reason.WORKER}:
            return BrowserState.RESULT
        return BrowserState.CLOSED

    def wait(self, timeout: float) -> bool:
        return self._ffp.wait(timeout=timeout)

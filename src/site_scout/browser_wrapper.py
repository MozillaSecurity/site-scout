# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from logging import getLogger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from .explorer import Explorer, ExplorerMode
    from .url import URL


LOG = getLogger(__name__)


@dataclass(eq=False, frozen=True)
class BrowserArgs:
    """Arguments specific to browser management.

    Attributes:
        binary: Browser binary.
        launch_timeout: Number of seconds to wait for browser to launch.
        memory_limit: Browser memory limit in bytes.
        debugger: Debugger to use.
        display_mode: Display mode to use.
        prefs_file: Custom prefs.js file to use.
        profile: Custom profile directory to use. TODO: UNUSED, remove?
    """

    binary: Path
    launch_timeout: int
    memory_limit: int
    debugger: str | None = None
    display_mode: str | None = None
    prefs_file: Path | None = None
    profile: Path | None = None


class BrowserState(Enum):
    """Possible browser states when complete."""

    RESULT = auto()
    CLOSED = auto()
    EXITED = auto()


class BrowserWrapper(ABC):
    """BrowserWrapper is used to add support for specific browsers."""

    __slots__ = ("args",)

    # pylint: disable=unused-argument
    def __init__(
        self, args: BrowserArgs, working_path: Path, env_mgr: EnvironmentManager | None
    ) -> None:
        self.args = args

    def __enter__(self) -> BrowserWrapper:
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup method. Remove all data at that is no longer required.

        Args:
            None

        Returns:
            None.
        """

    @abstractmethod
    def close(self) -> None:
        """Close target.

        Args:
            None

        Returns:
            None.
        """

    @abstractmethod
    def create_explorer(
        self, url: URL, load_wait: int, mode: ExplorerMode, pause: int = 0
    ) -> Explorer:
        """Create Explorer.

        Args:
            url: URL to navigate to.
            load_wait: Maximum number of seconds to wait for page load.
            mode: Operational mode.
            pause: Number of seconds to wait after page has loaded before continuing.

        Return:
            Explorer object.
        """

    @abstractmethod
    def debugger(self) -> bool:
        """Check if an external debugger is in use.

        Args:
            None

        Returns:
            True if a debugger is in use otherwise False.
        """

    @abstractmethod
    def dump_coverage(self, timeout: int = 0) -> None:
        """Trigger coverage data dump.

        Args:
            timeout: Amount of time to wait for data to be written.

        Returns:
            None.
        """

    @staticmethod
    @abstractmethod
    def environment_manager(
        instance_limit: int,
        browser_args: BrowserArgs,
    ) -> EnvironmentManager | None:
        """Create a custom environment manager instance.

        Args:
            instance_limit: Maximum number of browser instances to run in parallel.
            browser_args: Arguments passed to the browser.

        Returns:
            An environment manager object or None.
        """

    @abstractmethod
    def is_healthy(self) -> bool:
        """Check the browser is in a good state.

        Args:
            None

        Returns:
            True if the browser is running and determined to be
            in a valid functioning state otherwise False.
        """

    @abstractmethod
    def is_idle(self, idle_limit: int) -> bool:
        """Check if the browser is in an idle state.

        Args:
            idle_limit: Maximum CPU process usage to be considered idle.

        Returns:
            True if the browser is idle otherwise False.
        """

    @abstractmethod
    def launch(
        self,
        url: str,
        explore: bool,
        log_path: Path | None,
        raise_on_failure: bool,
    ) -> bool:
        """Launch the browser.

        Args:
            url: URL to load.
            explore: Use an Explorer to interact with the page.
            log_path: Location to save unexpected launch error logs.
            raise_on_failure: Raise or suppress launch a failure.

        Returns:
            True if the browser launched successfully otherwise False.
        """

    @abstractmethod
    def save_report(self, dst: Path) -> None:
        """Save logs and other related files to specified location.

        Args:
            dst: Location to save report.

        Returns:
            None.
        """

    @abstractmethod
    def state(self) -> BrowserState:
        """Browser state when complete. This is only intended to be used when the
        browser has executed and is no longer running. For example it indicates if the
        browser was closed, if it exited or if it crashed.

        Args:
            None.

        Returns:
            Browser state.
        """

    @abstractmethod
    def wait(self, timeout: float) -> bool:
        """Wait for a given amount of time while the browser is running.

        Args:
            timeout: Number of seconds to wait.

        Return:
            True if the browser exited before the timeout elapsed otherwise False.
        """


class EnvironmentManager(ABC):
    """EnvironmentManager is used to configure and control required components that are
    required to run browsers, for example, emulators."""

    @abstractmethod
    def cleanup(self) -> None:
        """Perform any necessary cleanup operations"""

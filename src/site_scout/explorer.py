# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import annotations

from enum import Enum, auto, unique
from logging import ERROR, WARNING, getLogger
from threading import Event, Lock, Thread
from time import perf_counter
from typing import TYPE_CHECKING

from page_explorer import ExplorerError, PageExplorer

if TYPE_CHECKING:
    from pathlib import Path

# suppress noise from page explorer deps
getLogger("selenium").setLevel(WARNING)
getLogger("urllib3").setLevel(ERROR)

LOG = getLogger(__name__)


@unique
class State(Enum):
    """State of Explorer"""

    PENDING = auto()
    CONNECTING = auto()
    NOT_FOUND = auto()
    LOADING = auto()
    EXPLORING = auto()
    CLOSING = auto()
    CLOSED = auto()


class ExplorerStatus:
    """Used to collect status data from PageExplorer."""

    __slots__ = ("explore_duration", "get_duration", "lock", "state")

    def __init__(self) -> None:
        self.explore_duration: float | None = None
        self.get_duration: float | None = None
        self.lock = Lock()
        self.state: State = State.PENDING


class Explorer:
    """PageExplorer wrapper to use enable use via SiteScout."""

    __slots__ = ("_can_skip", "_status", "_thread")

    def __init__(self, binary: Path, port: int, url: str) -> None:
        assert port is not None
        self._can_skip = Event()
        self._status = ExplorerStatus()
        self._thread = Thread(
            target=self._run,
            args=(binary, port, url, self._status, self._can_skip),
        )
        self._thread.start()

    def __enter__(self) -> Explorer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Unblock and join PageExplorer thread.

        Args:
            None

        Returns:
            None
        """
        self._can_skip.set()
        # this should NEVER timeout, if it does there is a bug
        self._thread.join(timeout=60)

    def explore_duration(self) -> float | None:
        """Amount of time in seconds spent exploring content.

        Args:
            None

        Returns:
            Time in seconds if available otherwise None.
        """
        with self._status.lock:
            return self._status.explore_duration

    def get_duration(self) -> float | None:
        """Amount of time in seconds spent loading content.

        Args:
            None

        Returns:
            Time in seconds if available otherwise None.
        """
        with self._status.lock:
            return self._status.get_duration

    def is_running(self) -> bool:
        """Check if PageExplorer thread is running.

        Args:
            None

        Returns:
            True if the thread is running otherwise False.
        """
        return self._thread.is_alive()

    def not_found(self) -> bool:
        """Check if server did not respond (server not found).

        Args:
            None

        Returns:
            True if the server did not respond otherwise False.
        """
        with self._status.lock:
            return self._status.state == State.NOT_FOUND

    def state(self) -> str:
        """Current state of the explorer.

        Args:
            None

        Returns:
            Current state.
        """
        with self._status.lock:
            return self._status.state.name

    @staticmethod
    def _run(
        binary: Path,
        port: int,
        url: str,
        status: ExplorerStatus,
        can_skip: Event,
    ) -> None:
        """Interact with content via PageExplorer.

        Args:
            binary: Binary launched.
            port: Browser driver listening port.
            url: Url to visit.
            status: Status tracker.
            can_skip: Used to delay execution.

        Returns:
            None.
        """

        def custom_wait(seconds: float) -> None:  # pragma: no cover
            assert seconds >= 0
            can_skip.wait(seconds)

        with status.lock:
            status.state = State.CONNECTING

        try:
            with PageExplorer(binary=binary, port=port) as explorer:
                with status.lock:
                    status.state = State.LOADING
                start_time = perf_counter()
                if not explorer.get(url):
                    # failed to navigate to web site
                    if explorer.is_connected():
                        with status.lock:
                            status.state = State.NOT_FOUND
                    return
                duration = perf_counter() - start_time
                with status.lock:
                    status.get_duration = duration
                    status.state = State.EXPLORING
                start_time = perf_counter()
                if not explorer.explore(wait_cb=custom_wait):
                    # failed to execute all explore instructions
                    return
                duration = perf_counter() - start_time
                with status.lock:
                    status.explore_duration = duration
                    status.state = State.CLOSING
                explorer.close_browser()
                with status.lock:
                    status.state = State.CLOSED
        except ExplorerError:
            LOG.debug("ExplorerError detected, aborting...")

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from logging import ERROR, WARNING, getLogger
from queue import Queue
from threading import Event, Thread
from time import perf_counter
from typing import TYPE_CHECKING

from page_explorer import ExplorerError, PageExplorer, PageLoad

if TYPE_CHECKING:
    from pathlib import Path

# suppress noise from page explorer deps
getLogger("selenium").setLevel(WARNING)
getLogger("urllib3").setLevel(ERROR)

LOG = getLogger(__name__)


class State(Enum):
    """State of Explorer"""

    INITIALIZING = auto()
    LOADING = auto()
    NOT_FOUND = auto()
    LOAD_FAILURE = auto()
    UNHANDLED_ERROR = auto()
    SKIP_CONTENT = auto()
    EXPLORING = auto()
    CLOSING = auto()
    CLOSED = auto()


PAGE_LOAD_FAILURES = frozenset(
    (
        State.NOT_FOUND,
        State.LOAD_FAILURE,
        State.UNHANDLED_ERROR,
    )
)


class ExplorerMode(Enum):
    """Used to control the operation of Explorer."""

    ALL = auto()
    LOAD = auto()


@dataclass(eq=False, frozen=True)
class ExplorerStatus:
    """Used to collect status data from PageExplorer."""

    explore_duration: float | None
    load_duration: float | None
    state: State
    url_loaded: str | None


class Explorer:
    """PageExplorer wrapper to enable use via SiteScout."""

    __slots__ = ("_can_skip", "_queue", "_status", "_thread")

    def __init__(
        self,
        binary: Path,
        port: int,
        url: str,
        mode: ExplorerMode = ExplorerMode.ALL,
        load_wait: int = 60,
        pause: int = 10,
    ) -> None:
        # init is used to wait for PageExplorer to connect
        init = Event()
        self._can_skip = Event()
        self._queue: Queue[ExplorerStatus] = Queue(maxsize=1)
        self._status: ExplorerStatus | None = None
        self._thread = Thread(
            target=self._run,
            args=(
                binary,
                port,
                url,
                mode,
                self._queue,
                self._can_skip,
                load_wait,
                pause,
                init,
            ),
        )
        self._thread.start()
        # wait for PageExplorer to connect before continuing
        if not init.wait(timeout=300):
            raise RuntimeError("PageExplorer thread did not unblock")

    def __enter__(self) -> Explorer:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Unblock and join PageExplorer thread and set status.

        Args:
            None

        Returns:
            None
        """
        # check if close() was already called
        if self._status is None:
            # disable delay to unblock shutdown
            self._can_skip.set()
            # this should NEVER timeout, if it does there is a bug
            self._status = self._queue.get(timeout=60)
            self._thread.join()

    def is_running(self) -> bool:
        """Check if PageExplorer thread is running.

        Args:
            None

        Returns:
            True if the thread is running otherwise False.
        """
        return self._thread.is_alive()

    @property
    def status(self) -> ExplorerStatus:
        """Final status of Explorer execution. Explorer.close() must be called first.

        Args:
            None

        Returns:
            Status of Explorer.
        """
        if self._status is None:
            raise RuntimeError("Explorer.close() not called")
        return self._status

    @staticmethod
    def _run(
        binary: Path,
        port: int,
        url: str,
        mode: ExplorerMode,
        queue: Queue[ExplorerStatus],
        can_skip: Event,
        load_wait: int,
        pause: int,
        init: Event,
    ) -> None:
        """Interact with content via PageExplorer.

        Args:
            binary: Binary launched.
            port: Browser driver listening port.
            url: Url to visit.
            mode: Operational mode.
            queue: Used to pass results.
            can_skip: Used to delay execution.
            load_wait: Time in seconds to wait for page to load before continuing.
            pause: Time in seconds to wait after load attempt is finished.
            init: Used to delay initialization of Explorer until PageExplorer is ready.

        Returns:
            None.
        """

        def _wait(seconds: float) -> None:  # pragma: no cover
            assert seconds >= 0
            can_skip.wait(seconds)

        explore_duration = None
        load_duration = None
        url_loaded = None
        state = State.INITIALIZING
        try:
            with PageExplorer(binary=binary, port=port) as explorer:
                # indicate PageExplorer has been initialized and connected to browser
                init.set()
                # attempt to navigate and load page
                state = State.LOADING
                start_time = perf_counter()
                get_result = explorer.get(url, wait=load_wait)
                title = explorer.title
                # verify page load
                if get_result == PageLoad.FAILURE:
                    if title == "Server Not Found":
                        state = State.NOT_FOUND
                    elif title == "Problem loading page":
                        state = State.LOAD_FAILURE
                    else:
                        state = State.UNHANDLED_ERROR
                    LOG.debug(
                        "load: %s (%0.1fs) - %s (%r)",
                        get_result.name,
                        perf_counter() - start_time,
                        url,
                        title,
                    )
                    # the browser is running but page cannot be loaded
                    return
                url_loaded = explorer.current_url
                load_duration = perf_counter() - start_time
                LOG.debug(
                    "load: %s (%0.1fs) - %r (%r)",
                    get_result.name,
                    load_duration,
                    title,
                    url_loaded,
                )
                # wait for more content to load/render/run
                LOG.debug("pausing for %ds...", pause)
                can_skip.wait(timeout=pause)
                if mode == ExplorerMode.ALL:
                    # attempt to find and activate "skip to content" link
                    state = State.SKIP_CONTENT
                    explorer.skip_to_content()
                    # interact with content
                    state = State.EXPLORING
                    start_time = perf_counter()
                    if not explorer.explore(wait_cb=_wait):
                        # failed to execute all explore instructions
                        return
                    explore_duration = perf_counter() - start_time
                # attempt to close the browser
                state = State.CLOSING
                explorer.close_browser(wait=10)
                if not explorer.is_connected():
                    state = State.CLOSED
        except ExplorerError:
            LOG.debug("ExplorerError detected, aborting...")
        finally:
            init.set()
            LOG.debug("final state: %s", state.name)
            queue.put(
                ExplorerStatus(explore_duration, load_duration, state, url_loaded)
            )

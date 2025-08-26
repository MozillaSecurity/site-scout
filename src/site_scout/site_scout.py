# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from itertools import chain
from json import dump
from logging import getLogger
from os import getenv

try:
    from os import getloadavg  # type: ignore[attr-defined]

    _LOAD_AVG = True
except ImportError:
    _LOAD_AVG = False

from pathlib import Path
from random import shuffle
from shutil import rmtree
from tempfile import gettempdir
from time import gmtime, perf_counter, sleep, strftime
from typing import TYPE_CHECKING

from ffpuppet import BrowserTimeoutError, Debugger, FFPuppet, LaunchError, Reason
from ffpuppet.display import DisplayMode

from .explorer import PAGE_LOAD_FAILURES, Explorer, ExplorerMode, State
from .reporter import FuzzManagerReporter
from .url import URL, URLParseError

if TYPE_CHECKING:
    from .url_db import UrlDB

LOG = getLogger(__name__)
TMP_PATH = Path(gettempdir()) / "site-scout"
TMP_PATH.mkdir(exist_ok=True, parents=True)


def trim(in_str: str, max_len: int) -> str:
    """Trim in_str if it exceeds max_len.

    Args:
        in_str: String to trim if required.
        max_len: Maximum length of output.

    Returns:
        Modified string or original string depending on max_len.
    """
    return f"{in_str[: max_len - 3]}..." if len(in_str) > max_len else in_str


# pylint: disable=too-many-instance-attributes
@dataclass(eq=False, frozen=True)
class VisitSummary:
    """Store data from completed Visit for analysis.

    Attributes:
        duration: Length of visit in seconds.
        identifier: Reference to location visited (URL or other handle).
        force_closed: Browser was forcibly closed.
        has_result: Indicates that results were detected.
        explore_duration: Time in seconds spent interacting with content.
        load_duration: Time in seconds spent loading content.
        state: Enum representing visit progress.
        url_collection: An optional label to track origin of URL.
        url_loaded: URL actually visited.
    """

    duration: float
    identifier: str
    force_closed: bool
    has_result: bool
    explore_duration: float | None = None
    load_duration: float | None = None
    state: State | None = None
    url_collection: str | None = None
    url_loaded: str | None = None


class Status:
    """Track and report status to a file."""

    __slots__ = ("_dst", "_next", "_rate_limit", "_start")

    def __init__(self, dst: Path, rate_limit: int = 60) -> None:
        assert dst
        assert rate_limit >= 0
        self._dst = dst
        self._next: float = 0
        self._rate_limit = rate_limit
        self._start = perf_counter()

    def report(
        self,
        active: int,
        jobs: int,
        completed: int,
        target: int,
        results: int,
        not_found: int,
        load_failure: int,
        avg_duration: int = 0,
        include_rate: bool = False,
        force: bool = False,
    ) -> None:
        """Write status report to the filesystem.

        Args:
            active: Number of active browsers.
            jobs: Maximum number of active browsers.
            completed: URLs opened.
            target: Total URLs to be opened.
            results: Number of results found.
            not_found: Number of 'server not found' errors.
            load_failure: Number of page load failures.
            avg_duration: Average page visit duration.
            include_rate: Include visit rate.
            force: Ignore rate limit and report.

        Returns:
            None
        """
        now = perf_counter()
        if not force and self._next > now:
            return
        comp_pct = (completed / target) * 100 if target else 0.0
        runtime = now - self._start
        with self._dst.open("w") as lfp:
            lfp.write(f" Active/Limit : {active}/{jobs}\n")
            lfp.write(f"Current/Total : {completed}/{target} ({comp_pct:0.1f}%)\n")
            lfp.write(f"      Results : {results}\n")
            if not_found > 0:
                nf_pct = (not_found / completed) * 100
                lfp.write(f"    Not Found : {not_found} ({nf_pct:0.1f}%)\n")
            if load_failure > 0:
                lf_pct = (load_failure / completed) * 100
                lfp.write(f"Load Failures : {load_failure} ({lf_pct:0.1f}%)\n")
            if avg_duration > 0:
                lfp.write(f" Avg Duration : {avg_duration}s\n")
            if include_rate:
                rate = int(completed / (runtime / 60 / 60)) if runtime else 0
                lfp.write(f"   Visit Rate : {rate}/h\n")
            if _LOAD_AVG:
                sys_load = ", ".join(f"{x:0.2f}" for x in getloadavg())
                lfp.write(f"  System Load : {sys_load}\n")
            lfp.write(f"      Runtime : {timedelta(seconds=int(runtime))}\n")
            lfp.write(f"    Timestamp : {strftime('%Y/%m/%d %X %z', gmtime())}\n")
        self._next = now + self._rate_limit


class Visit:
    """Visit contains details about the site and browser."""

    __slots__ = (
        "_end_time",
        "_start_time",
        "explorer",
        "idle_timestamp",
        "puppet",
        "url",
    )

    def __init__(self, puppet: FFPuppet, url: URL, explorer: Explorer | None) -> None:
        self._end_time: float | None = None
        self._start_time = perf_counter()
        self.explorer = explorer
        self.idle_timestamp: float | None = None
        self.puppet = puppet
        self.url = url

    def cleanup(self) -> None:
        """Close dependents and cleanup.

        Args:
            None

        Returns:
            None.
        """
        self.close()
        self.puppet.clean_up()

    def close(self) -> None:
        """Close browser and explorer if needed.

        Args:
            None

        Returns:
            None.
        """
        if self._end_time is None:
            self._end_time = perf_counter()
            # close browser before closing explorer
            self.puppet.close()
            if self.explorer is not None:
                self.explorer.close()

    def duration(self) -> float:
        """Total runtime of the visit.

        Args:
            None

        Returns:
            Visit duration.
        """
        if self._end_time is not None:
            return self._end_time - self._start_time
        return perf_counter() - self._start_time

    def summary(self) -> VisitSummary:
        """Create VisitSummary from Visit data.

        NOTE: If URL.alias is set no URL data should be passed to the VisitSummary.

        Args:
            None.

        Returns:
            VisitSummary.
        """
        assert not self.is_active()
        has_result = self.puppet.reason in frozenset((Reason.ALERT, Reason.WORKER))
        force_closed = self.puppet.reason == Reason.CLOSED
        url_collection = getenv("URL_COLLECTION")
        if self.explorer is not None:
            return VisitSummary(
                self.duration(),
                self.url.alias or str(self.url),
                force_closed,
                has_result,
                explore_duration=self.explorer.status.explore_duration,
                load_duration=self.explorer.status.load_duration,
                state=self.explorer.status.state,
                url_collection=url_collection,
                url_loaded=None if self.url.alias else self.explorer.status.url_loaded,
            )
        return VisitSummary(
            self.duration(),
            self.url.alias or str(self.url),
            force_closed,
            has_result,
            url_collection=url_collection,
        )

    def is_active(self) -> bool:
        """Check if visit is in progress.

        Args:
            None

        Returns:
            True if visit is in progress otherwise False.
        """
        return self._end_time is None


# pylint: disable=too-many-instance-attributes
class SiteScout:
    """SiteScout can visit a collection of URLs and report process failures."""

    __slots__ = (
        "_active",
        "_binary",
        "_complete",
        "_coverage",
        "_debugger",
        "_display_mode",
        "_explore",
        "_fuzzmanager",
        "_launch_failure_limit",
        "_launch_failures",
        "_launch_timeout",
        "_log_limit",
        "_memory_limit",
        "_omit_urls",
        "_prefs",
        "_profile",
        "_summaries",
        "_urls",
    )

    def __init__(
        self,
        binary: Path,
        profile: Path | None = None,
        prefs_js: Path | None = None,
        debugger: Debugger = Debugger.NONE,
        display_mode: str = "default",
        launch_timeout: int = 180,
        launch_failure_limit: int = 3,
        log_limit: int = 0,
        memory_limit: int = 0,
        explore: str | None = None,
        fuzzmanager: bool = False,
        coverage: bool = False,
        omit_urls: bool = False,
    ) -> None:
        assert launch_failure_limit > 0
        self._active: list[Visit] = []
        self._complete: list[Visit] = []
        self._summaries: list[VisitSummary] = []
        self._urls: list[URL] = []
        # browser related
        self._binary = binary
        self._coverage = coverage
        self._debugger = debugger
        self._display_mode = display_mode
        self._explore = ExplorerMode[explore.upper()] if explore else None
        self._launch_failure_limit = launch_failure_limit
        # consecutive launch failures
        self._launch_failures = 0
        self._launch_timeout = launch_timeout
        self._log_limit = log_limit
        self._memory_limit = memory_limit
        self._omit_urls = omit_urls
        self._prefs = prefs_js
        self._profile = profile
        # reporter
        self._fuzzmanager: FuzzManagerReporter | None = (
            FuzzManagerReporter(binary, working_path=TMP_PATH) if fuzzmanager else None
        )

    def __enter__(self) -> SiteScout:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close and cleanup browser instances.

        Args:
            None

        Returns:
            None
        """
        LOG.debug("closing %d active visit(s)...", len(self._active))
        for visit in chain(self._active, self._complete):
            visit.cleanup()
        self._active.clear()
        self._complete.clear()

    def _launch(self, url: URL, log_path: Path | None = None) -> bool:
        """Launch a new browser instance and visit provided URL.

        Args:
            url: URL to visit.
            log_path: Directory to save launch failure logs.

        Returns:
            True if the browser was launched otherwise False.
        """
        success = False
        ffp = FFPuppet(
            debugger=self._debugger,
            display_mode=DisplayMode[self._display_mode.upper()],
            use_profile=self._profile,
            working_path=str(TMP_PATH),
        )
        try:
            ffp.launch(
                self._binary,
                env_mod={"MOZ_CRASHREPORTER_SHUTDOWN": "1"},
                location=None if self._explore else str(url),
                launch_timeout=self._launch_timeout,
                log_limit=self._log_limit,
                marionette=0 if self._explore else None,
                memory_limit=self._memory_limit,
                prefs_js=self._prefs,
            )
            explorer: Explorer | None = None
            if self._explore:
                assert ffp.marionette is not None
                # this can raise RuntimeError
                explorer = Explorer(
                    self._binary,
                    ffp.marionette,
                    str(url),
                    mode=self._explore,
                    load_wait=60 if self._debugger == Debugger.NONE else 90,
                    pause=10,
                )
            self._active.append(Visit(ffp, url, explorer=explorer))
            self._launch_failures = 0
            success = True
        except LaunchError as exc:
            self._launch_failures += 1
            is_failure = not isinstance(exc, BrowserTimeoutError)
            LOG.warning("Browser launch %s...", "failure" if is_failure else "timeout")
            if self._launch_failures >= self._launch_failure_limit:
                # save failure
                if is_failure and log_path is not None:
                    ffp.close()
                    dst = log_path / strftime("%Y%m%d-%H%M%S-launch-failure")
                    ffp.save_logs(dst)
                    LOG.warning("Logs saved '%s'", dst)
                raise
        finally:
            # cleanup if launch was unsuccessful
            if not success:
                ffp.clean_up()
        return success

    def load_db(self, data: UrlDB) -> None:
        """Load URLs from a UrlDB and add them to the queue.
        NOTE: This should only be used with known valid URL data.

        Args:
            data: Dictionary containing URLs.

        Returns:
            None
        """
        total_domains = 0
        total_subdomains = 0
        total_urls = 0
        LOG.debug("processing dict...")
        existing = {x.uid for x in self._urls}
        while data:
            domain, subdomains = data.popitem()
            total_domains += 1
            for subdomain, paths in subdomains.items():
                total_subdomains += 1
                for path in paths:
                    url = URL(domain, subdomain=subdomain, path=path)
                    # avoid duplicates
                    if url.uid not in existing:
                        if self._omit_urls:
                            url.alias = "REDACTED"
                        self._urls.append(url)
                        existing.add(url.uid)
                    total_urls += 1
        LOG.debug(
            "%d domain(s), %d subdomain(s), %d URL(s) processed, %d loaded",
            total_domains,
            total_subdomains,
            total_urls,
            len(existing),
        )

    def load_str(self, url: str, alias: str | None = None) -> None:
        """Parse, sanitize and add a URL to the queue.

        Args:
            url: Location to visit.
            alias: Value to use as alias if available.

        Returns:
            None
        """
        try:
            parsed = URL.parse(url)
        except URLParseError as exc:
            # only show full error message when omit_urls is false
            LOG.error("Failed to parse URL%s", "" if self._omit_urls else f": {exc}")
            return
        # add unique urls to queue
        # NOTE: this might get slow with large lists
        if parsed.uid not in {x.uid for x in self._urls}:
            if alias:
                parsed.alias = alias
            elif self._omit_urls:
                parsed.alias = "REDACTED"
            self._urls.append(parsed)

    # pylint: disable=too-many-branches
    def _process_active(
        self,
        time_limit: int,
        idle_usage: int = 10,
        idle_wait: int = 10,
        min_visit: int = 20,
    ) -> None:
        """Check active browser processes and move complete visits from the active
        to the complete queue.

        Args:
            time_limit: Maximum time to wait for a site to load.
            idle_usage: Maximum CPU usage to be considered idle.
            idle_wait: Amount of time in seconds process must be idle.
            min_visit: Delay in seconds before idle checks begin.

        Returns:
            None
        """
        assert time_limit > 0
        assert min_visit > 0
        assert idle_usage >= 0
        assert idle_wait >= 0

        complete: list[int] = []
        for index, visit in enumerate(self._active):
            # check if work is complete
            visit_runtime = visit.duration()
            if not visit.puppet.is_healthy():
                visit.close()
                complete.append(index)
            # check if explorer is complete
            elif visit.explorer and not visit.explorer.is_running():
                LOG.debug("explorer not running (%s)", visit.url.uid[:6])
                # call explorer.close() so when can check final state
                visit.explorer.close()
                # check if browser closed or potentially crashed
                if visit.explorer.status.state not in PAGE_LOAD_FAILURES:
                    visit.puppet.wait(10 if self._debugger == Debugger.NONE else 30)
                visit.close()
                complete.append(index)
            elif visit_runtime >= time_limit:
                LOG.debug("visit timeout (%s)", visit.url.uid[:6])
                if self._coverage:
                    visit.puppet.dump_coverage()
                visit.close()
                complete.append(index)
            # check for idle browser instance
            elif idle_usage and visit_runtime >= min_visit:
                # check all browser processes are below idle limit
                if all(x[1] < idle_usage for x in visit.puppet.cpu_usage()):
                    now = perf_counter()
                    if visit.idle_timestamp is None:
                        LOG.debug("set idle (%s)", visit.url.uid[:6])
                        visit.idle_timestamp = now
                    if now - visit.idle_timestamp >= idle_wait:
                        LOG.debug("visit idle (%s)", visit.url.uid[:6])
                        if self._coverage:
                            visit.puppet.dump_coverage()
                        visit.close()
                        complete.append(index)
                elif visit.idle_timestamp is not None:
                    LOG.debug("reset idle (%s)", visit.url.uid[:6])
                    visit.idle_timestamp = None

        if complete:
            for index in sorted(complete, reverse=True):
                self._complete.append(self._active.pop(index))
            LOG.debug("%d active, %d removed", len(self._active), len(complete))

    def _process_complete(self, log_path: Path) -> int:
        """Report results, record summaries and remove completed visits.

        Args:
            log_path: Directory to save results in.

        Returns:
            Number of results found.
        """
        results = 0
        while self._complete:
            LOG.debug("%d complete visit(s) to process", len(self._complete))
            # collect summary
            # NOTE: ALL reported data should be derived from summary
            visit = self._complete.pop()
            summary = visit.summary()
            self._summaries.append(summary)
            # process and report result
            if summary.has_result:
                # WARNING: START REPORTING/SAVING DATA
                dst = log_path / strftime(f"%Y%m%d-%H%M%S-result-{visit.url.uid[:6]}")
                visit.puppet.save_logs(dst)
                results += 1
                LOG.info(
                    "Result found visiting '%s' (%0.1fs)",
                    summary.identifier,
                    summary.duration,
                )
                # save prefs file
                if self._prefs:
                    (dst / "prefs.js").write_text(self._prefs.read_text())
                # collect visit metadata
                metadata = {
                    "duration": f"{summary.duration:0.1f}",
                    "identifier": summary.identifier,
                }
                if summary.state is not None:
                    metadata["explore_state"] = summary.state.name
                if summary.url_loaded is not None:
                    metadata["url_loaded"] = summary.url_loaded
                if summary.url_collection is not None:
                    metadata["url_collection"] = summary.url_collection
                with (dst / "metadata.json").open("w") as ofp:
                    dump(metadata, ofp, indent=2, sort_keys=True)
                # report result
                if self._fuzzmanager:
                    fm_id, short_sig = self._fuzzmanager.submit(dst, metadata)
                    LOG.info("FuzzManager (%d): %s", fm_id, trim(short_sig, 60))
                    # remove local data when reporting to FM
                    rmtree(dst)
                else:
                    LOG.info("Saved as '%s'", dst)
                # DONE REPORTING/SAVING DATA

            # handle page load failures
            if summary.state == State.LOAD_FAILURE:
                LOG.info("Page load failure: '%s'", summary.identifier)
            elif summary.state == State.NOT_FOUND:
                LOG.info("Server Not Found: '%s'", summary.identifier)
                self._skip_url(visit.url, state=State.NOT_FOUND)
            elif summary.state == State.UNHANDLED_ERROR:
                LOG.info("Unexpected load error: %s", summary.identifier)
            visit.cleanup()
        return results

    def schedule_urls(
        self, url_limit: int = 0, randomize: bool = True, visits: int = 1
    ) -> None:
        """Prepare URL queue. Randomize and limit size as needed.

        Args:
            url_limit: Limit total URLs when value is greater than zero.
            shuffle_urls: Randomize order of URLs in the queue.
            visits: Number of times to visit each URL.

        Returns:
            None
        """
        assert url_limit >= 0
        if randomize:
            # shuffle before enforcing limit so all entries can potentially be included
            shuffle(self._urls)
        if url_limit and len(self._urls) > url_limit:
            LOG.info("Enforcing URL limit (%d -> %d)", len(self._urls), url_limit)
            self._urls = self._urls[:url_limit]
        if not randomize:
            # provided url list is processed in reverse order
            # reverse the list to maintain original order
            self._urls.reverse()
        if visits > 1:
            LOG.info("Each URL will be visited %dx", visits)
            # repeat the list for multiple visits
            self._urls = self._urls * visits

    def _skip_remaining(self) -> None:
        """Skip remaining visits. Clear the URL queue, close and cleanup active browser
        instances.

        Args:
            None

        Returns:
            None
        """
        self._urls.clear()
        LOG.debug("skipping active visits: %d", len(self._active))
        for visit in self._active:
            visit.cleanup()
        self._active.clear()

    def _skip_url(self, url: URL, state: State | None = None) -> None:
        """Remove URLs with matching domain and subdomain from the queue.

        Args:
            url: Used as filter.
            state: Explore state to set on skipped.

        Returns:
            None.
        """
        removed = 0
        for idx in reversed(range(len(self._urls))):
            if (
                self._urls[idx].domain == url.domain
                and self._urls[idx].subdomain == url.subdomain
            ):
                url = self._urls.pop(idx)
                self._summaries.append(
                    VisitSummary(0, url.alias or str(url), False, False, state=state)
                )
                removed += 1
        if removed > 0:
            LOG.info("Skipping %d related queued URLs", removed)

    def run(
        self,
        log_path: Path,
        time_limit: int,
        check_delay: int = 1,
        domain_rate_limit: int = 20,
        instance_limit: int = 1,
        status_report: Path | None = None,
        result_limit: int = 0,
        runtime_limit: int = 0,
    ) -> None:
        """Iterate over the queue and visit each URL. Each visit is performed by a
        new browser instance with a clean profile.

        Args:
            log_path: Location to write results.
            time_limit: Maximum time in seconds of a site visit.
            check_delay: Time in seconds between checking for results.
            domain_rate_limit: Minimum time in seconds between visiting the same domain.
            instance_limit: Maximum number of browser sessions to run at once.
            status_report: File to populate with status report data.
            result_limit: Number of results that can be found before exiting.
            runtime_limit: Maximum number of seconds to run.

        Returns:
            None
        """
        assert check_delay >= 0
        assert domain_rate_limit >= 0
        assert instance_limit > 0
        assert result_limit >= 0
        assert runtime_limit >= 0

        if runtime_limit > 0:
            minutes, seconds = divmod(runtime_limit, 60)
            hours, minutes = divmod(minutes, 60)
            LOG.info("Runtime limit is %02d:%02d:%02d", hours, minutes, seconds)

        end_time = int(perf_counter() + runtime_limit) if runtime_limit > 0 else 0
        last_visit: dict[str, float] = {}
        status = Status(status_report) if status_report else None
        total_results = 0
        total_urls = len(self._urls)
        while self._urls or self._active:
            # perform status report
            if status:
                status.report(
                    len(self._active),
                    instance_limit,
                    len(self._summaries),
                    total_urls,
                    total_results,
                    sum(1 for x in self._summaries if x.state == State.NOT_FOUND),
                    sum(1 for x in self._summaries if x.state == State.LOAD_FAILURE),
                )

            # select url to visit and launch browser
            if self._urls and (len(self._active) < instance_limit):
                next_url = self._urls.pop()
                # avoid frequent domain visits by rate limiting
                if (
                    next_url.domain in last_visit
                    and perf_counter() - last_visit[next_url.domain] < domain_rate_limit
                ):
                    LOG.debug(
                        "domain rate limit hit (%s)", next_url.alias or next_url.domain
                    )
                    # move url to the end of the queue
                    self._urls.insert(0, next_url)
                # attempt to launch browser and visit url
                elif self._launch(next_url, log_path=log_path):
                    short_url = trim(next_url.alias or str(next_url), 80)
                    LOG.info(
                        "[%02d/%02d] '%s'",
                        len(self._active) + len(self._summaries),
                        total_urls,
                        short_url,
                    )
                    LOG.debug(
                        "launched, explore: %s, timeout: %ds, %s - %s",
                        self._explore.name if self._explore else None,
                        time_limit,
                        next_url.uid[:6],
                        short_url,
                    )
                    last_visit[next_url.domain] = perf_counter()
                    assert self._active
                # launch failed
                else:
                    # re-add url to queue since it was not visited
                    self._urls.append(next_url)

            # check for complete processes (disable idle checks when explore is set)
            self._process_active(time_limit, idle_usage=0 if self._explore else 10)
            total_results += self._process_complete(log_path=log_path)

            # check result and runtime limits
            if result_limit and self._urls and total_results >= result_limit:
                LOG.info("Result limit (%d) hit", result_limit)
                self._skip_remaining()
                assert not self._active
                assert not self._urls
            elif 0 < end_time <= perf_counter():
                LOG.info("Runtime limit (%ds) hit", runtime_limit)
                self._skip_remaining()
                assert not self._active
                assert not self._urls

            # wait a moment for work to complete
            if self._urls or self._active:
                sleep(check_delay)

        # final status report
        if status:
            avg_duration = (
                int(
                    sum(
                        # only include "successful" page loads in calculation
                        x.duration
                        for x in self._summaries
                        if x.state not in PAGE_LOAD_FAILURES
                    )
                    / len(self._summaries)
                )
                if self._summaries
                else 0
            )

            status.report(
                len(self._active),
                instance_limit,
                len(self._summaries),
                total_urls,
                total_results,
                sum(1 for x in self._summaries if x.state == State.NOT_FOUND),
                sum(1 for x in self._summaries if x.state == State.LOAD_FAILURE),
                avg_duration=avg_duration,
                include_rate=True,
                force=True,
            )
        LOG.info(
            "Visits complete: %d, results: %d", len(self._summaries), total_results
        )

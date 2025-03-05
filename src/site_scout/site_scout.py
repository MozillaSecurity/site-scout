# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from hashlib import sha1
from itertools import chain
from logging import getLogger
from pathlib import Path
from random import shuffle
from re import compile as re_compile
from shutil import rmtree
from string import punctuation
from tempfile import gettempdir
from time import gmtime, perf_counter, sleep, strftime
from typing import Any, NewType
from urllib.parse import quote, urlsplit

from ffpuppet import BrowserTimeoutError, Debugger, FFPuppet, LaunchError, Reason
from ffpuppet.display import DisplayMode

from .explorer import Explorer
from .reporter import FuzzManagerReporter

LOG = getLogger(__name__)
TMP_PATH = Path(gettempdir()) / "site-scout"
TMP_PATH.mkdir(exist_ok=True, parents=True)


# This is used as a placeholder for empty subdomains
# WARNING: If this changes all yml files will need to be updated
NO_SUBDOMAIN = "*"

# Note: Python 3.10+ use TypeAlias
UrlDB = NewType("UrlDB", dict[str, dict[str, list[str]]])


def trim(in_str: str, max_len: int) -> str:
    """Trim in_str if it exceeds max_len.

    Args:
        in_str: String to trim if required.
        max_len: Maximum length of output.

    Returns:
        Modified string or original string depending on max_len.
    """
    return f"{in_str[:max_len - 3]}..." if len(in_str) > max_len else in_str


# pylint: disable=too-many-instance-attributes
@dataclass(eq=False)
class VisitSummary:
    """Store data from completed Visits for analysis."""

    duration: float
    url: URL
    load_duration: float | None = None
    explore_duration: float | None = None
    explore_state: str | None = None
    force_closed: bool = False
    has_result: bool = False
    not_found: bool = False


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
        avg_duration: int,
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
            force: Ignore rate limit and report.

        Returns:
            None
        """
        now = perf_counter()
        if not force and self._next > now:
            return
        comp_pct = (completed / target) * 100 if target else 0.0
        with self._dst.open("w") as lfp:
            lfp.write(f" Active/Limit : {active}/{jobs}\n")
            lfp.write(f"Current/Total : {completed}/{target} ({comp_pct:0.1f}%)\n")
            lfp.write(f"      Results : {results}\n")
            if not_found > 0:
                lfp.write(f"    Not Found : {not_found}\n")
            if avg_duration > 0:
                lfp.write(f" Avg Duration : {avg_duration}s\n")
            lfp.write(f"      Runtime : {timedelta(seconds=int(now - self._start))}\n")
            lfp.write(f"    Timestamp : {strftime('%Y/%m/%d %X %z', gmtime())}\n")
        self._next = now + self._rate_limit


class URL:
    """URL components."""

    ALLOWED_SCHEMES = frozenset(("http", "https"))
    VALID_DOMAIN = re_compile(r"[a-zA-Z0-9_.-]")

    __slots__ = ("_uid", "domain", "path", "scheme", "subdomain")

    def __init__(
        self,
        domain: str,
        subdomain: str | None = None,
        path: str = "/",
        scheme: str = "http",
    ) -> None:
        self.domain = domain
        self.path = path
        self.scheme = scheme
        self.subdomain = subdomain
        self._uid: str | None = None

    def __str__(self) -> str:
        if self.subdomain is None or self.subdomain == NO_SUBDOMAIN:
            return f"{self.scheme}://{self.domain}{self.path}"
        return f"{self.scheme}://{self.subdomain}.{self.domain}{self.path}"

    # pylint: disable=too-many-return-statements
    @classmethod
    def create(
        cls,
        domain: str,
        subdomain: str | None = None,
        path: str = "/",
        scheme: str = "http",
    ) -> URL | None:
        """Sanitize, verify data and create a URL if possible.

        Args:
            domain: Domain.
            subdomain: Subdomain.
            path: Path, must begin with '/'.
            scheme: Scheme.

        Returns:
            URL object if input is valid otherwise None.
        """
        scheme = scheme.lower()
        if scheme not in cls.ALLOWED_SCHEMES:
            LOG.error("Cannot create URL: Invalid scheme %r", scheme)
            return None

        try:
            # use idna to encode domain with non ascii characters
            domain = domain.lower().encode("idna").decode("ascii")
        except UnicodeError:
            LOG.error("Cannot create URL: Invalid domain %r", domain)
            return None
        if not cls.VALID_DOMAIN.match(domain):
            LOG.error("Cannot create URL: Invalid domain %r", domain)
            return None

        if subdomain is not None and subdomain != NO_SUBDOMAIN:
            if not subdomain:
                LOG.error("Cannot create URL: Empty subdomain")
                return None
            try:
                subdomain = subdomain.lower().encode("idna").decode("ascii")
            except UnicodeError:
                LOG.error("Cannot create URL: Invalid subdomain %r", subdomain)
                return None
            if not cls.VALID_DOMAIN.match(subdomain):
                LOG.error("Cannot create URL: Invalid subdomain %r", subdomain)
                return None

        if not path.startswith("/"):
            LOG.error("Cannot create URL: Path must begin with '/'")
            return None

        # percent encode non ascii characters in path if needed
        if not path.isascii():
            path = quote(path, safe=punctuation)

        return cls(domain, subdomain=subdomain, path=path, scheme=scheme)

    @property
    def uid(self) -> str:
        """Unique ID.

        Args:
            None

        Returns:
            Unique ID.
        """
        # this does NOT need to be cryptographically secure
        # it needs to be filesystem safe and *somewhat* unique
        if not self._uid:
            self._uid = sha1(
                str(self).encode(errors="replace"), usedforsecurity=False
            ).hexdigest()
        return self._uid


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
                LOG.debug("%s explorer: %s", self.url.uid[:6], self.explorer.state())
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
        "_cert_files",
        "_complete",
        "_coverage",
        "_debugger",
        "_display_mode",
        "_explore",
        "_extension",
        "_fuzzmanager",
        "_launch_failure_limit",
        "_launch_failures",
        "_launch_timeout",
        "_log_limit",
        "_memory_limit",
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
        extension: list[Path] | None = None,
        cert_files: list[Path] | None = None,
        explore: bool = False,
        fuzzmanager: bool = False,
        coverage: bool = False,
    ) -> None:
        assert launch_failure_limit > 0
        self._active: list[Visit] = []
        self._complete: list[Visit] = []
        self._summaries: list[VisitSummary] = []
        self._urls: list[URL] = []
        # browser related
        self._binary = binary
        self._cert_files = cert_files
        self._coverage = coverage
        self._debugger = debugger
        self._display_mode = display_mode
        self._explore = explore
        self._extension = extension
        self._launch_failure_limit = launch_failure_limit
        # consecutive launch failures
        self._launch_failures = 0
        self._launch_timeout = launch_timeout
        self._log_limit = log_limit
        self._memory_limit = memory_limit
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
                extension=self._extension,
                cert_files=self._cert_files,
            )
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
        else:
            explorer: Explorer | None = None
            if self._explore:
                assert ffp.marionette is not None
                # this can raise RuntimeError
                explorer = Explorer(self._binary, ffp.marionette, str(url))
            self._active.append(Visit(ffp, url, explorer=explorer))
            self._launch_failures = 0
        finally:
            # cleanup if launch was unsuccessful
            if self._launch_failures != 0:
                ffp.clean_up()
        return self._launch_failures == 0

    def load_dict(self, data: UrlDB) -> None:
        """Load URLs from a UrlDB and add them to the queue.

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
                    url = URL(
                        domain,
                        subdomain=subdomain if subdomain != NO_SUBDOMAIN else None,
                        path=path,
                    )
                    # avoid duplicates
                    if url.uid not in existing:
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

    def load_str(self, url: str) -> None:
        """Parse, sanitize and add a URL to the queue.

        Args:
            url: Location to visit.

        Returns:
            None
        """
        assert url
        if "://" not in url:
            url = f"http://{url}"
        parsed = urlsplit(url, allow_fragments=False)
        if parsed.scheme not in URL.ALLOWED_SCHEMES:
            LOG.error("Unsupported scheme in URL: %r", url)
            return None
        path = parsed.path if parsed.path else "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        # this currently does not separate domain and subdomain
        formatted = URL.create(parsed.netloc, path=path, scheme=parsed.scheme)
        # add unique urls to queue
        # NOTE: this might get slow with large lists
        if formatted is not None and formatted.uid not in {x.uid for x in self._urls}:
            self._urls.append(formatted)
        return None

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
            elif visit_runtime >= time_limit:
                LOG.debug("visit timeout (%s)", visit.url.uid[:6])
                if self._coverage:
                    visit.puppet.dump_coverage()
                visit.close()
                complete.append(index)
            # check if explorer is complete but browser is running
            elif visit.explorer and not visit.explorer.is_running():
                LOG.debug("explorer not running (%s)", visit.url.uid[:6])
                if not visit.explorer.not_found():
                    # pause in case browser is closing (debuggers and slow builds)
                    visit.puppet.wait(30)
                visit.close()
                complete.append(index)
            # check all browser processes are below idle limit
            elif idle_usage and visit_runtime >= min_visit:
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
            visit = self._complete.pop()
            assert not visit.is_active()
            duration = visit.duration()
            summary = VisitSummary(duration, visit.url)

            if visit.puppet.reason in (Reason.ALERT, Reason.WORKER):
                summary.has_result = True
                dst = log_path / strftime(f"%Y%m%d-%H%M%S-result-{visit.url.uid[:6]}")
                visit.puppet.save_logs(dst)
                if self._prefs:
                    (dst / "prefs.js").write_text(self._prefs.read_text())
                (dst / "duration.txt").write_text(f"{duration:0.1f}")
                (dst / "url.txt").write_text(str(visit.url))
                results += 1
                LOG.info("Result found visiting '%s' (%0.1fs)", visit.url, duration)
                if self._fuzzmanager:
                    metadata = {
                        "duration": f"{duration:0.1f}",
                        "url": str(visit.url),
                    }
                    if visit.explorer is not None:
                        metadata["explore_state"] = visit.explorer.state()
                        if visit.explorer.url_loaded:
                            metadata["url_loaded"] = visit.explorer.url_loaded
                    fm_id, short_sig = self._fuzzmanager.submit(dst, metadata)
                    LOG.info("FuzzManager (%d): %s", fm_id, trim(short_sig, 60))
                    # remove local data when reporting to FM
                    rmtree(dst)
                else:
                    LOG.info("Saved as '%s'", dst)
            else:
                summary.force_closed = visit.puppet.reason != Reason.EXITED

            if visit.explorer is not None:
                summary.explore_duration = visit.explorer.explore_duration()
                summary.explore_state = visit.explorer.state()
                summary.load_duration = visit.explorer.load_duration()
                summary.not_found = visit.explorer.not_found()
                if summary.not_found:
                    LOG.info("Server Not Found: '%s'", visit.url)
                    self._skip_not_found(visit.url.domain)

            self._summaries.append(summary)
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

    def _skip_not_found(self, domain: str) -> None:
        """Remove URLs with matching domain from the queue.

        Args:
            domain: Value used as filter.

        Returns:
            None.
        """
        removed = 0
        for idx in reversed(range(len(self._urls))):
            if self._urls[idx].domain == domain:
                url = self._urls.pop(idx)
                self._summaries.append(VisitSummary(0, url, not_found=True))
                removed += 1
        if removed > 0:
            LOG.info("Skipping %d related queued URLs", removed)

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

    # pylint: disable=too-many-locals,too-many-statements
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
                    sum(1 for x in self._summaries if x.not_found),
                    0,
                )

            # select url to visit and launch browser
            if self._urls and (len(self._active) < instance_limit):
                next_url = self._urls.pop()
                # avoid frequent domain visits by rate limiting
                if (
                    next_url.domain in last_visit
                    and perf_counter() - last_visit[next_url.domain] < domain_rate_limit
                ):
                    LOG.debug("domain rate limit hit (%s)", next_url.domain)
                    # move url to the end of the queue
                    self._urls.insert(0, next_url)
                # attempt to launch browser and visit url
                elif self._launch(next_url, log_path=log_path):
                    short_url = trim(str(next_url), 80)
                    LOG.info(
                        "[%02d/%02d] %r",
                        len(self._active) + len(self._summaries),
                        total_urls,
                        short_url,
                    )
                    LOG.debug(
                        "launched, explore: %r, timeout: %ds, %s - %s",
                        self._explore,
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
            if self._summaries:
                # don't include "server not found" results in calculation
                avg_duration = int(
                    sum(x.duration for x in self._summaries if not x.not_found)
                    / len(self._summaries)
                )
            else:
                avg_duration = 0

            status.report(
                len(self._active),
                instance_limit,
                len(self._summaries),
                total_urls,
                total_results,
                sum(1 for x in self._summaries if x.not_found),
                avg_duration,
                force=True,
            )
        LOG.info(
            "Visits complete: %d, results: %d", len(self._summaries), total_results
        )


# pylint: disable=too-many-return-statements
def verify_dict(data: Any, allow_empty: bool = False) -> str | None:
    """Verify the structure of data.

    Args:
        data: Dictionary to check.
        allow_empty: Empty data set is valid if True.

    Return:
        An error message is returned if a problem is found.
    """
    if not isinstance(data, dict):
        return "Invalid data"
    if not data and not allow_empty:
        return "No data found"
    # check domains
    for domain, subdomains in data.items():
        if not isinstance(domain, str) or not domain:
            return "Domain must be a string"
        if not isinstance(subdomains, dict) or not subdomains:
            return f"Invalid domain entry: '{domain}'"
        # check subdomains
        for subdomain, paths in subdomains.items():
            if not isinstance(subdomain, str) or not subdomain:
                return "Subdomain must be a string"
            if not isinstance(paths, list) or not paths:
                return f"Invalid subdomain entry: '{subdomain}' in '{domain}'"
            # check paths
            for path in paths:
                if not isinstance(path, str) or not path.startswith("/"):
                    return "Path must be a string starting with '/'"
    return None

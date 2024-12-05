# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import annotations

from datetime import timedelta
from hashlib import sha1
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
        force: bool = False,
    ) -> None:
        """Write status report to the filesystem.

        Args:
            active: Number of active browsers.
            jobs: Maximum number of active browsers.
            completed: URLs opened.
            target: Total URLs to be opened.
            results: Number of results found.
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
            lfp.write(f"      Runtime : {timedelta(seconds=int(now - self._start))}\n")
            lfp.write(f"    Timestamp : {strftime('%Y/%m/%d %X %z', gmtime())}\n")
        self._next = now + self._rate_limit


class URL:
    """URL components."""

    ALLOWED_SCHEMES = frozenset(("http", "https"))
    VALID_DOMAIN = re_compile(r"[a-zA-Z0-9_.-]")

    __slots__ = ("domain", "scheme", "subdomain", "path", "_uid")

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
        if self.subdomain is None:
            return f"{self.scheme}://{self.domain}{self.path}"
        return f"{self.scheme}://{self.subdomain}.{self.domain}{self.path}"

    @staticmethod
    def is_ascii(data: str) -> bool:
        """Check if all characters in a string are ASCII.

        Args:
            data: String to check.

        Returns:
            True if string only contains ASCII characters otherwise False.
        """
        return all(ord(x) < 128 for x in data)

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

        # use idna to encode domain with non ascii characters
        domain = domain.lower().encode("idna").decode("ascii")
        if not cls.VALID_DOMAIN.match(domain):
            LOG.error("Cannot create URL: Invalid domain %r", domain)
            return None

        if subdomain is not None and subdomain != NO_SUBDOMAIN:
            if not subdomain:
                LOG.error("Cannot create URL: Empty subdomain")
                return None
            subdomain = subdomain.lower().encode("idna").decode("ascii")
            if not cls.VALID_DOMAIN.match(subdomain):
                LOG.error("Cannot create URL: Invalid subdomain %r", subdomain)
                return None

        if not path.startswith("/"):
            LOG.error("Cannot create URL: Path must begin with '/'")
            return None

        # percent encode non ascii characters in path if needed
        if not cls.is_ascii(path):
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
            self._uid = sha1(str(self).encode(errors="replace")).hexdigest()
        return self._uid


class Visit:
    """Visit contains details about the site and browser."""

    __slots__ = ("end_time", "idle_timestamp", "puppet", "url", "start_time")

    def __init__(self, puppet: FFPuppet, url: URL, start_time: float) -> None:
        self.end_time: float | None = None
        self.idle_timestamp: float | None = None
        self.puppet = puppet
        self.url = url
        self.start_time = start_time


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
        "_display",
        "_extension",
        "_fuzzmanager",
        "_launch_failure_limit",
        "_launch_failures",
        "_launch_timeout",
        "_log_limit",
        "_memory_limit",
        "_prefs",
        "_profile",
        "_urls",
    )

    def __init__(
        self,
        binary: Path,
        profile: Path | None = None,
        prefs_js: Path | None = None,
        debugger: Debugger = Debugger.NONE,
        display: str | None = None,
        launch_timeout: int = 180,
        launch_failure_limit: int = 3,
        log_limit: int = 0,
        memory_limit: int = 0,
        extension: list[Path] | None = None,
        cert_files: list[Path] | None = None,
        fuzzmanager: bool = False,
        coverage: bool = False,
    ) -> None:
        assert launch_failure_limit > 0
        self._active: list[Visit] = []
        self._complete: list[Visit] = []
        self._urls: list[URL] = []
        # browser related
        self._binary = binary
        self._cert_files = cert_files
        self._coverage = coverage
        self._debugger = debugger
        self._display = display
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
        for visit in self._active:
            visit.puppet.clean_up()
        self._active.clear()
        for visit in self._complete:
            visit.puppet.clean_up()
        self._complete.clear()

    def _launch(self, url: URL, log_path: Path | None = None) -> bool:
        """Launch a new browser instance and visit provided URL.

        Args:
            url: URL to visit.
            log_path: Directory to save launch failure logs.

        Returns:
            True if the browser was launched otherwise False.
        """
        env_mod: dict[str, str | None] = {"MOZ_CRASHREPORTER_SHUTDOWN": "1"}

        ffp = FFPuppet(
            debugger=self._debugger,
            headless=self._display,
            use_profile=self._profile,
            working_path=str(TMP_PATH),
        )
        try:
            ffp.launch(
                self._binary,
                env_mod=env_mod,
                location=str(url),
                launch_timeout=self._launch_timeout,
                log_limit=self._log_limit,
                memory_limit=self._memory_limit,
                prefs_js=self._prefs,
                extension=self._extension,
                cert_files=self._cert_files,
            )
            self._launch_failures = 0
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
            if self._launch_failures != 0:
                ffp.clean_up()

        if self._launch_failures == 0:
            self._active.append(Visit(ffp, url, perf_counter()))
        return self._launch_failures == 0

    def load_dict(self, data: UrlDB) -> None:
        """Load URLs from a UrlDB (dict) and add to list of URLs to visit.

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
        """Parse, sanitize and add a URL to list of URLs to visit.

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
            raise ValueError(f"Unsupported scheme in URL: {parsed.scheme}")
        path = parsed.path if parsed.path else "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        # this currently does not separate domain and subdomain
        formatted = URL.create(parsed.netloc, path=path, scheme=parsed.scheme)
        assert formatted is not None
        # this might get slow with large lists
        if formatted.uid not in {x.uid for x in self._urls}:
            self._urls.append(formatted)

    def _process_active(
        self,
        time_limit: float,
        idle_usage: int = 10,
        idle_wait: float = 10.0,
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
            visit_runtime = perf_counter() - visit.start_time
            if not visit.puppet.is_healthy():
                visit.end_time = perf_counter()
                visit.puppet.close()
                complete.append(index)
            elif visit_runtime >= time_limit:
                LOG.debug("visit timeout (%s)", visit.url.uid[:8])
                if self._coverage:
                    visit.puppet.dump_coverage()
                visit.end_time = perf_counter()
                visit.puppet.close()
                complete.append(index)
            # check all browser processes are below idle limit
            elif idle_usage and visit_runtime >= min_visit:
                if all(x[1] < idle_usage for x in visit.puppet.cpu_usage()):
                    now = perf_counter()
                    if visit.idle_timestamp is None:
                        LOG.debug("set idle (%s)", visit.url.uid[:8])
                        visit.idle_timestamp = now
                    if now - visit.idle_timestamp >= idle_wait:
                        LOG.debug("visit idle (%s)", visit.url.uid[:8])
                        if self._coverage:
                            visit.puppet.dump_coverage()
                        visit.end_time = perf_counter()
                        visit.puppet.close()
                        complete.append(index)
                elif visit.idle_timestamp is not None:
                    LOG.debug("reset idle (%s)", visit.url.uid[:8])
                    visit.idle_timestamp = None

        if complete:
            LOG.debug("found %d complete", len(complete))
            for index in sorted(complete, reverse=True):
                self._complete.append(self._active.pop(index))
            LOG.debug("%d active, %d complete", len(self._active), len(self._complete))

    def _process_complete(self, log_path: Path) -> int:
        """Report results and remove completed visits.

        Args:
            log_path: Directory to save results in.

        Returns:
            Number of results found.
        """
        results = 0
        while self._complete:
            LOG.debug("%d pending visit(s) to check", len(self._complete))
            visit = self._complete.pop()
            assert visit.end_time is not None
            if visit.puppet.reason in (Reason.ALERT, Reason.WORKER):
                dst = log_path / strftime(f"%Y%m%d-%H%M%S-result-{visit.url.uid[:8]}")
                visit.puppet.save_logs(dst)
                duration = visit.end_time - visit.start_time
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
                    fm_id, short_sig = self._fuzzmanager.submit(dst, metadata)
                    LOG.info("FuzzManager (%d): %s", fm_id, trim(short_sig, 60))
                    # remove local data when reporting to FM
                    rmtree(dst)
                else:
                    LOG.info("Saved as '%s'", dst)
            visit.puppet.clean_up()
        return results

    def schedule_urls(
        self, url_limit: int = 0, randomize: bool = True, visits: int = 1
    ) -> None:
        """Prepare URL list. Randomize and limit size as needed.

        Args:
            url_limit: Limit total URLs when value is greater than zero.
            shuffle_urls: Randomly order URLs visits.
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
        """Skip remaining visits. This will clear the backlog of remaining Visits and
        close and cleanup active browser instances.

        Args:
            None

        Returns:
            None
        """
        self._urls.clear()
        LOG.debug("skipping active visits: %d", len(self._active))
        for visit in self._active:
            visit.puppet.clean_up()
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
        """Iterate over and visit each URL. Each visit is performed in a new browser
        instance using a clean profile.

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
        total_visits = 0
        while self._urls or self._active:
            # perform status report
            if status:
                status.report(
                    len(self._active),
                    instance_limit,
                    total_visits,
                    total_urls,
                    total_results,
                )

            # select url to visit and launch browser
            if self._urls and (len(self._active) < instance_limit):
                next_url = self._urls.pop()
                # avoid frequent domain visits by rate limiting
                if (
                    perf_counter() - last_visit.get(next_url.domain, 0)
                ) < domain_rate_limit:
                    LOG.debug("domain rate limit hit (%s)", next_url.domain)
                    self._urls.insert(0, next_url)
                # launch browser and visit url
                elif self._launch(next_url, log_path=log_path):
                    short_url = trim(str(next_url), 80)
                    total_visits += 1
                    LOG.info("[%02d/%02d] %r", total_visits, total_urls, short_url)
                    LOG.debug(
                        "launched browser visiting [%s] %s",
                        next_url.uid[:8],
                        short_url,
                    )
                    last_visit[next_url.domain] = perf_counter()
                    assert self._active

            # check for complete processes
            self._process_active(time_limit)
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
            status.report(
                len(self._active),
                instance_limit,
                total_visits,
                total_urls,
                total_results,
                force=True,
            )
        LOG.info("URL visits %d, results reported %d", total_visits, total_results)


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

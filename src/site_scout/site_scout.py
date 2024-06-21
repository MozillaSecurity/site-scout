# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from datetime import timedelta
from hashlib import sha1
from logging import getLogger
from pathlib import Path
from random import shuffle
from shutil import rmtree
from tempfile import gettempdir
from time import gmtime, sleep, strftime, time
from typing import Any, Dict, List, NewType, Optional
from urllib.parse import urlsplit

from ffpuppet import BrowserTimeoutError, Debugger, FFPuppet, LaunchError, Reason

from .reporter import FuzzManagerReporter

LOG = getLogger(__name__)
TMP_PATH = Path(gettempdir()) / "site-scout"
TMP_PATH.mkdir(exist_ok=True, parents=True)


# This is used as a placeholder for empty subdomains
# WARNING: If this changes all yml files will need to be updated
NO_SUBDOMAIN = "*"

# Note: Python 3.10+ use TypeAlias
UrlDB = NewType("UrlDB", Dict[str, Dict[str, List[str]]])


class Status:
    """Track and report status to a file."""

    __slots__ = ("_dst", "_next", "_rate_limit", "_start")

    def __init__(self, dst: Path, rate_limit: int = 60) -> None:
        assert dst
        assert rate_limit >= 0
        self._dst = dst
        self._next: float = 0
        self._rate_limit = rate_limit
        self._start = time()

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
        now = time()
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

    __slots__ = ("domain", "scheme", "subdomain", "path", "_uid")

    def __init__(
        self,
        domain: str,
        subdomain: Optional[str] = None,
        path: str = "/",
        scheme: str = "http",
    ) -> None:
        assert domain
        assert path.startswith("/")
        assert subdomain is None or subdomain
        self.domain = domain.lower()
        self.scheme = scheme.lower()
        assert self.scheme in self.ALLOWED_SCHEMES
        self.subdomain = subdomain.lower() if subdomain else None
        self.path = path
        self._uid: Optional[str] = None

    def __str__(self) -> str:
        if self.subdomain is None:
            return f"{self.scheme}://{self.domain}{self.path}"
        return f"{self.scheme}://{self.subdomain}.{self.domain}{self.path}"

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

    __slots__ = ("idle_timestamp", "puppet", "url", "timestamp")

    def __init__(
        self, puppet: FFPuppet, url: URL, timestamp: Optional[float] = None
    ) -> None:
        self.idle_timestamp: Optional[float] = None
        self.puppet = puppet
        self.url = url
        self.timestamp = time() if timestamp is None else timestamp
        assert self.timestamp >= 0


# pylint: disable=too-many-instance-attributes
class SiteScout:
    """SiteScout can visit a collection of URLs and report process failures."""

    __slots__ = (
        "_active",
        "_binary",
        "_cert_files",
        "_complete",
        "_debugger",
        "_display",
        "_extension",
        "_fuzzmanager",
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
        profile: Optional[Path] = None,
        prefs_js: Optional[Path] = None,
        debugger: Debugger = Debugger.NONE,
        display: Optional[str] = None,
        launch_timeout: int = 180,
        log_limit: int = 0,
        memory_limit: int = 0,
        extension: Optional[List[Path]] = None,
        cert_files: Optional[List[Path]] = None,
        fuzzmanager: bool = False,
    ) -> None:
        self._active: List[Visit] = []
        self._complete: List[Visit] = []
        self._urls: List[URL] = []
        # browser related
        self._binary = binary
        self._cert_files = cert_files
        self._debugger = debugger
        self._extension = extension
        self._display = display
        self._launch_timeout = launch_timeout
        self._log_limit = log_limit
        self._memory_limit = memory_limit
        self._prefs = prefs_js
        self._profile = profile
        # reporter
        self._fuzzmanager: Optional[FuzzManagerReporter] = (
            FuzzManagerReporter(binary, working_path=TMP_PATH) if fuzzmanager else None
        )

    def __enter__(self) -> "SiteScout":
        return self

    def __exit__(self, *exc: Any) -> None:
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

    def _launch(
        self, url: URL, launch_attempts: int = 3, log_path: Optional[Path] = None
    ) -> None:
        """Launch a new browser instance and visit provided URL.

        Args:
            url: URL to visit.
            launch_attempts: Attempts to launch the browser before raising.
            log_path: Directory to save launch failure logs.

        Returns:
            None
        """
        assert launch_attempts > 0

        env_mod: Dict[str, Optional[str]] = {"MOZ_CRASHREPORTER_SHUTDOWN": "1"}

        success = False
        for attempt in range(1, launch_attempts + 1):
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
                success = True
            except LaunchError as exc:
                is_failure = not isinstance(exc, BrowserTimeoutError)
                LOG.warning(
                    "Browser launch %s (attempt %d/%d)",
                    "failure" if is_failure else "timeout",
                    attempt,
                    launch_attempts,
                )
                if attempt == launch_attempts:
                    # save failure
                    if is_failure and log_path is not None:
                        ffp.close()
                        dst = log_path / strftime("%Y%m%d-%H%M%S-launch-failure")
                        ffp.save_logs(dst)
                        LOG.warning("Logs saved '%s'", dst)
                    raise
            finally:
                if not success:
                    ffp.clean_up()
            if success:
                break
            # launch attempt limit not met... retry
            sleep(1)

        self._active.append(Visit(ffp, url))

    def load_dict(self, data: UrlDB) -> None:
        """Load URLs from a dict add to list of URLs to visit.

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
        if "://" in url:
            scheme = url.split("://", maxsplit=1)[0]
            if scheme.lower() not in URL.ALLOWED_SCHEMES:
                raise ValueError(f"Unsupported scheme in URL: {scheme}")
        else:
            url = f"http://{url}"
        parsed = urlsplit(url, allow_fragments=False)
        path = parsed.path if parsed.path else "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        # this currently does not separate domain and subdomain
        formatted = URL(parsed.netloc, path=path, scheme=parsed.scheme)
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

        complete: List[int] = []
        for index, visit in enumerate(self._active):
            # check if work is complete
            visit_runtime = time() - visit.timestamp
            if not visit.puppet.is_healthy():
                visit.puppet.close()
                complete.append(index)
            elif visit_runtime >= time_limit:
                LOG.debug("visit timeout (%s)", visit.url.uid[:8])
                visit.puppet.close()
                complete.append(index)
            # check all browser processes are below idle limit
            elif idle_usage and visit_runtime >= min_visit:
                if all(x[1] < idle_usage for x in visit.puppet.cpu_usage()):
                    now = time()
                    if visit.idle_timestamp is None:
                        LOG.debug("set idle (%s)", visit.url.uid[:8])
                        visit.idle_timestamp = now
                    if now - visit.idle_timestamp >= idle_wait:
                        LOG.debug("visit idle (%s)", visit.url.uid[:8])
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
            if visit.puppet.reason in (Reason.ALERT, Reason.WORKER):
                dst = log_path / strftime(f"%Y%m%d-%H%M%S-result-{visit.url.uid[:8]}")
                visit.puppet.save_logs(dst)
                if self._prefs:
                    (dst / "prefs.js").write_text(self._prefs.read_text())
                (dst / "url.txt").write_text(str(visit.url))
                results += 1
                LOG.info("Result found visiting '%s'", visit.url)
                if self._fuzzmanager:
                    fm_id, short_sig = self._fuzzmanager.submit(dst)
                    LOG.info(
                        "FuzzManager (%d): %s",
                        fm_id,
                        f"{short_sig[:50]}..." if len(short_sig) > 50 else short_sig,
                    )
                    # remove local data when reporting to FM
                    rmtree(dst)
                else:
                    LOG.info("Saved as '%s'", dst)
            visit.puppet.clean_up()
        return results

    def schedule_urls(self, url_limit: int = 0, randomize: bool = True) -> None:
        """Prepare URL list. Randomize and limit size as needed.

        Args:
            url_limit: Limit total URLs when value is greater than zero.
            shuffle_urls: Randomly order URLs visits.

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

    # pylint: disable=too-many-locals
    def run(
        self,
        log_path: Path,
        time_limit: int,
        check_delay: float = 1.0,
        domain_rate_limit: int = 20,
        instance_limit: int = 1,
        status_report: Optional[Path] = None,
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

        Returns:
            None
        """
        assert check_delay >= 0
        assert domain_rate_limit >= 0
        assert instance_limit > 0

        last_visit: Dict[str, float] = {}
        next_url: Optional[URL] = None
        status = Status(status_report) if status_report else None
        total_results = 0
        total_urls = len(self._urls)
        total_visits = 0
        while self._urls or self._active:
            assert not next_url

            # perform status report
            if status:
                status.report(
                    len(self._active),
                    instance_limit,
                    total_visits,
                    total_urls,
                    total_results,
                )

            # select url to visit
            if self._urls and (len(self._active) < instance_limit):
                next_url = self._urls.pop()
                # avoid hammering specific domains by using rate limit
                if time() - last_visit.get(next_url.domain, 0) < domain_rate_limit:
                    LOG.debug("domain rate limit hit (%s)", next_url.domain)
                    self._urls.insert(0, next_url)
                    next_url = None

            # launch browser and visit url
            if next_url:
                url_str = str(next_url)
                total_visits += 1
                LOG.info(
                    "[%02d/%02d] %r",
                    total_visits,
                    total_urls,
                    f"{url_str[:80]}..." if len(url_str) > 80 else url_str,
                )
                self._launch(next_url, log_path=log_path)
                last_visit[next_url.domain] = time()
                next_url = None
                assert self._active

            # check for complete processes
            self._process_active(time_limit)
            total_results += self._process_complete(log_path=log_path)

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
        LOG.info("URLs visited %d, results reported %d", total_visits, total_results)


# pylint: disable=too-many-return-statements
def verify_dict(data: Any, allow_empty: bool = False) -> Optional[str]:
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

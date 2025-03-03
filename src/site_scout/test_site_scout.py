# type: ignore
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
# pylint: disable=protected-access
from itertools import chain, count, cycle

from ffpuppet import BrowserTerminatedError, LaunchError, Reason
from pytest import mark, raises

from .site_scout import NO_SUBDOMAIN, URL, SiteScout, Status, Visit, verify_dict


def test_url_str():
    """test URL.__str__()"""
    assert str(URL("a.c", subdomain="b", path="/d")) == "http://b.a.c/d"
    assert str(URL("a.c")) == str(URL("a.c", subdomain=NO_SUBDOMAIN))


@mark.parametrize(
    "domain, subdomain, path, scheme, expected",
    [
        # with a subdomain
        ("a.c", "b", "/d", "http", "http://b.a.c/d"),
        # without a subdomain
        ("a.c", None, "/d", "http", "http://a.c/d"),
        # with NO_SUBDOMAIN subdomain place holder
        ("a.c", NO_SUBDOMAIN, "/d", "http", "http://a.c/d"),
        # more complex
        ("a.c:1337", "x.y", "/1/2/3.html", "https", "https://x.y.a.c:1337/1/2/3.html"),
        # normalize case (domain and subdomain only)
        ("aB.cD", "Ab", "/Ef", "HTTP", "http://ab.ab.cd/Ef"),
        # normalize case (missing subdomain)
        ("aB.cD", None, "/Ef", "HTTP", "http://ab.cd/Ef"),
        # idna encode domain
        ("é.c", "ò", "/", "http", "http://xn--jda.xn--9ca.c/"),
        # percent encode non ascii chars path
        ("a.c", None, "/ñ%41=1", "http", "http://a.c/%C3%B1%41=1"),
        # previously encoded path
        ("a.c", None, "/%C3%B1o", "http", "http://a.c/%C3%B1o"),
    ],
)
def test_url_create(domain, subdomain, path, scheme, expected):
    """test URL.create()"""
    url = URL.create(domain, subdomain=subdomain, path=path, scheme=scheme)
    assert str(url) == expected
    assert url.uid


@mark.parametrize(
    "domain, subdomain, path, scheme",
    [
        # bad domain
        ("*", None, "/", "http"),
        # bad domain
        ("", None, "/", "http"),
        # bad domain
        (".a.c", None, "/", "http"),
        # bad subdomain
        ("a.c", f"{NO_SUBDOMAIN}.foo", "/", "http"),
        # bad subdomain
        ("a.c", "$.foo", "/", "http"),
        # bad subdomain
        ("a.c", "", "/d", "http"),
        # bad subdomain
        ("a.c", "..a", "/", "http"),
        # bad path
        ("a.c", None, "", "http"),
        # bad path
        ("a.c", None, "foo", "http"),
        # bad scheme
        ("a.c", None, "/", "foo"),
    ],
)
def test_url_create_failures(domain, subdomain, path, scheme):
    """test URL.create()"""
    assert URL.create(domain, subdomain=subdomain, path=path, scheme=scheme) is None


@mark.parametrize("explore", [False, True])
def test_visit_basic(mocker, explore):
    """test Visit"""
    exp = mocker.patch("site_scout.site_scout.Explorer", autospec=True).return_value
    puppet = mocker.patch("site_scout.site_scout.FFPuppet", autospec=True).return_value
    visit = Visit(puppet, URL("foo"), explorer=exp if explore else None)
    assert visit.is_active()
    assert visit.duration() != visit.duration()
    assert visit.puppet == puppet
    assert puppet.close.call_count == 0
    assert puppet.clean_up.call_count == 0
    if explore:
        assert visit.explorer == exp
        assert exp.close.call_count == 0
    else:
        assert visit.explorer is None
    visit.close()
    assert not visit.is_active()
    assert visit.duration() == visit.duration()
    assert puppet.close.call_count == 1
    assert puppet.clean_up.call_count == 0
    if explore:
        assert exp.close.call_count == 1
    visit.cleanup()
    assert puppet.close.call_count == 1
    assert puppet.clean_up.call_count == 1
    if explore:
        assert exp.close.call_count == 1


@mark.parametrize("explore", [False, True])
def test_site_scout_launch(mocker, explore):
    """test SiteScout._launch()"""
    mocker.patch("site_scout.site_scout.Explorer", autospec=True)
    mocker.patch("site_scout.site_scout.FFPuppet", autospec=True)
    with SiteScout(None, explore=explore) as scout:
        assert not scout._active
        assert scout._launch(URL("someurl"))
        assert scout._active
        assert scout._active[0].is_active()
        assert scout._active[0].idle_timestamp is None
        assert scout._active[0].puppet is not None
        assert str(scout._active[0].url) == "http://someurl/"
        if explore:
            assert scout._active[0].explorer is not None
        else:
            assert scout._active[0].explorer is None


@mark.parametrize("max_failures", [1, 2])
def test_site_scout_launch_failues(mocker, tmp_path, max_failures):
    """test SiteScout._launch() failures"""
    ffp = mocker.patch("site_scout.site_scout.FFPuppet", autospec=True)
    ffp.return_value.launch.side_effect = BrowserTerminatedError()
    with SiteScout(None) as scout:
        scout._launch_failure_limit = max_failures
        if max_failures > 1:
            assert not scout._launch("http://a/")
            assert ffp.return_value.save_logs.call_count == 0
        else:
            with raises(BrowserTerminatedError):
                scout._launch("http://a/", log_path=tmp_path)
            assert ffp.return_value.save_logs.call_count == 1
        assert ffp.return_value.clean_up.call_count == 1


def test_site_scout_close(mocker):
    """test SiteScout.close()"""
    active = mocker.Mock(spec_set=Visit)
    complete = mocker.Mock(spec_set=Visit)
    with SiteScout(None, explore=True) as scout:
        scout._active = [active]
        scout._complete = [complete]
        scout.close()
        assert not scout._active
        assert not scout._complete
    assert active.cleanup.call_count == 1
    assert complete.cleanup.call_count == 1


@mark.parametrize(
    "urls, is_healthy, timeout, cpu_usage, idle, active, explore",
    [
        # no urls to process
        ([], True, False, None, False, 0, False),
        # one active
        ([URL("foo")], True, False, None, False, 1, False),
        # multiple active
        ([URL("foo"), URL("bar")], True, False, None, False, 2, False),
        # one complete
        ([URL("foo")], False, False, None, False, 0, False),
        # timeout
        ([URL("foo")], True, True, None, False, 0, False),
        # idle
        ([URL("foo")], True, False, 0, False, 0, False),
        # reset idle
        ([URL("foo")], True, False, 100, True, 1, False),
        # explorer fails to close browser
        ([URL("foo")], True, False, None, False, 0, True),
    ],
)
def test_site_scout_process_active(
    mocker, urls, is_healthy, timeout, cpu_usage, idle, active, explore
):
    """test SiteScout._process_active()"""
    explorer = mocker.patch("site_scout.site_scout.Explorer", autospec=True)
    explorer.return_value.is_running.return_value = False
    explorer.return_value.not_found.return_value = False
    ffpuppet = mocker.patch("site_scout.site_scout.FFPuppet", autospec=True)
    ffpuppet.return_value.is_healthy.return_value = is_healthy
    ffpuppet.return_value.cpu_usage.return_value = [(None, cpu_usage)]
    with SiteScout(None, coverage=True, explore=explore) as scout:
        assert not scout._active
        assert scout._coverage
        for url in urls:
            scout._launch(url)
        total_active = len(scout._active)
        # setup state
        if timeout:
            for visit in scout._active:
                visit._start_time = 0
        if cpu_usage is not None:
            for visit in scout._active:
                visit._start_time -= 10
                if idle:
                    visit.idle_timestamp = visit._start_time
        # run and verify
        scout._process_active(30, idle_usage=10, idle_wait=0, min_visit=5)
        assert len(scout._active) == active
        assert active or total_active == len(scout._complete)
        for entry in scout._active:
            assert entry.is_active()
            assert not entry.puppet.close.call_count
        for entry in scout._complete:
            assert not entry.is_active()
            assert entry.puppet.close.call_count
            if is_healthy and not explore:
                assert entry.puppet.dump_coverage.call_count
            else:
                assert not entry.puppet.dump_coverage.call_count


@mark.parametrize(
    "urls, reason, reports",
    [
        # no urls to process
        ([], None, 0),
        # interesting result
        ([URL("foo")], Reason.ALERT, 1),
        # multiple interesting results
        ([URL("foo"), URL("bar")], Reason.ALERT, 2),
        # uninteresting result
        ([URL("foo")], Reason.CLOSED, 0),
        # uninteresting result
        ([URL("foo")], Reason.EXITED, 0),
        # interesting result
        ([URL("foo")], Reason.WORKER, 1),
    ],
)
def test_site_scout_process_complete(mocker, tmp_path, urls, reason, reports):
    """test SiteScout._process_complete()"""

    # pylint: disable=unused-argument
    def save_logs(dst_path, logs_only=False):
        dst_path.mkdir(exist_ok=True)

    explorer = mocker.patch("site_scout.site_scout.Explorer", autospec=True)
    ffpuppet = mocker.patch("site_scout.site_scout.FFPuppet", autospec=True)
    ffpuppet.return_value.is_healthy.return_value = False
    ffpuppet.return_value.reason = reason
    ffpuppet.return_value.save_logs.side_effect = save_logs
    prefs = tmp_path / "prefs.js"
    prefs.touch()
    report_dst = tmp_path / "reports"
    report_dst.mkdir()
    with SiteScout(None, prefs_js=prefs, explore=True) as scout:
        assert not scout._active
        for url in urls:
            scout._launch(url)
        assert scout._active or not urls
        scout._process_active(30)
        assert not scout._active
        assert len(scout._complete) == len(urls)
        assert not scout._summaries
        assert scout._process_complete(report_dst) == reports
        assert len(scout._summaries) == len(urls)
    assert sum(1 for _ in report_dst.iterdir()) == reports
    assert explorer.return_value.close.call_count == len(urls)


@mark.parametrize(
    "reason, explore",
    [
        (Reason.ALERT, True),
        (Reason.EXITED, True),
        (Reason.EXITED, False),
    ],
)
def test_site_scout_process_complete_summaries(mocker, tmp_path, reason, explore):
    """test SiteScout._process_complete() summaries"""

    # pylint: disable=unused-argument
    def save_logs(dst_path, logs_only=False):
        dst_path.mkdir(exist_ok=True)

    explorer = mocker.patch("site_scout.site_scout.Explorer", autospec=True)
    explorer.return_value.load_duration.return_value = 1.0
    explorer.return_value.explore_duration.return_value = 2.0
    explorer.return_value.state.return_value = "STATE"
    ffpuppet = mocker.patch("site_scout.site_scout.FFPuppet", autospec=True)
    ffpuppet.return_value.is_healthy.return_value = False
    ffpuppet.return_value.reason = reason
    ffpuppet.return_value.save_logs.side_effect = save_logs

    with SiteScout(None, explore=explore) as scout:
        assert not scout._active
        scout._launch(URL("foo"))
        assert len(scout._active) == 1
        scout._process_active(30)
        assert not scout._active
        assert len(scout._complete) == 1
        assert not scout._summaries
        scout._process_complete(tmp_path)
        assert len(scout._summaries) == 1
        assert scout._summaries[0].duration > 0
        assert scout._summaries[0].url.domain == "foo"
        assert not scout._summaries[0].force_closed
        assert scout._summaries[0].has_result == (reason == Reason.ALERT)
        if explore:
            assert scout._summaries[0].explore_state == "STATE"
            assert scout._summaries[0].load_duration == 1.0
            assert scout._summaries[0].explore_duration == 2.0
        else:
            assert scout._summaries[0].explore_state is None
            assert scout._summaries[0].load_duration is None
            assert scout._summaries[0].explore_duration is None


@mark.parametrize(
    "urls, reason, jobs, reports, use_fm, status, explore, result_limit, runtime_limit",
    [
        # no urls to process
        ([], None, 1, 0, False, True, False, 0, 0),
        # interesting result
        ([URL("foo")], Reason.ALERT, 1, 1, False, False, False, 0, 0),
        # job > work
        ([URL("foo")], Reason.ALERT, 2, 1, False, False, False, 0, 0),
        # multiple interesting results
        ([URL("foo"), URL("bar")], Reason.ALERT, 1, 2, False, False, False, 0, 0),
        # work > jobs
        (
            [URL("1"), URL("2"), URL("3"), URL("4")],
            Reason.ALERT,
            2,
            4,
            False,
            False,
            False,
            0,
            0,
        ),
        # uninteresting result
        ([URL("foo")], Reason.CLOSED, 1, 0, False, False, False, 0, 0),
        # domain rate limit
        ([URL("foo"), URL("foo")], Reason.CLOSED, 1, 0, False, False, False, 0, 0),
        # timeout
        ([URL("foo")], None, 1, 0, False, False, False, 0, 0),
        # report via FuzzManager
        ([URL("foo")], Reason.ALERT, 1, 1, True, False, False, 0, 0),
        # report via FuzzManager (explore)
        ([URL("foo")], Reason.ALERT, 1, 1, True, False, True, 0, 0),
        # report status
        ([URL("foo")], Reason.ALERT, 1, 1, False, True, False, 0, 0),
        # hit result limit
        ([URL("foo"), URL("bar")], Reason.ALERT, 1, 1, False, False, False, 1, 0),
        # hit runtime limit
        ([URL("foo"), URL("bar")], None, 1, 0, False, False, False, 0, 1),
        # explore
        ([URL("foo"), URL("bar")], None, 1, 0, False, False, True, 0, 0),
    ],
)
def test_site_scout_run(
    mocker,
    tmp_path,
    urls,
    reason,
    jobs,
    reports,
    use_fm,
    status,
    explore,
    result_limit,
    runtime_limit,
):  # pylint: disable=too-many-locals
    """test SiteScout.run()"""

    # pylint: disable=unused-argument
    def save_logs(dst_path, logs_only=False):
        dst_path.mkdir(exist_ok=True)

    mocker.patch("site_scout.site_scout.Explorer", autospec=True)
    mocker.patch("site_scout.site_scout.sleep", autospec=True)
    mocker.patch("site_scout.site_scout.perf_counter", side_effect=count())
    ffpuppet = mocker.patch("site_scout.site_scout.FFPuppet", autospec=True)
    reporter = mocker.patch("site_scout.site_scout.FuzzManagerReporter", autospec=True)
    reporter.return_value.submit.return_value = (1337, "[@ sig]")
    if reason:
        # only first pass is running
        ffpuppet.return_value.is_healthy.side_effect = chain([True], cycle([False]))
    ffpuppet.return_value.reason = reason
    ffpuppet.return_value.save_logs.side_effect = save_logs

    report_dst = tmp_path / "reports"
    report_dst.mkdir()
    status_file = tmp_path / "status.txt" if status else None
    with SiteScout(None, explore=explore, fuzzmanager=use_fm) as scout:
        assert not scout._active
        scout._urls = urls
        scout.run(
            report_dst,
            10,
            instance_limit=jobs,
            status_report=status_file,
            result_limit=result_limit,
            runtime_limit=runtime_limit,
        )
        assert len(scout._active) + len(scout._complete) == len(urls)
        for visit in chain(scout._active, scout._complete):
            if explore:
                assert visit.explorer is not None
            else:
                assert visit.explorer is None
    assert sum(1 for _ in report_dst.iterdir()) == (0 if use_fm else reports)
    assert reporter.return_value.submit.call_count == (reports if use_fm else 0)


def test_site_scout_run_launch_failed(mocker, tmp_path):
    """test SiteScout.run() launch failed"""
    mocker.patch("site_scout.site_scout.sleep", autospec=True)
    mocker.patch("site_scout.site_scout.perf_counter", side_effect=count())
    ffpuppet = mocker.patch("site_scout.site_scout.FFPuppet", autospec=True)
    # one launch failure and one successful launch
    ffpuppet.return_value.launch.side_effect = (LaunchError("foo"), None)
    with SiteScout(None, explore=False) as scout:
        assert not scout._active
        scout._urls = [URL("test")]
        scout.run(tmp_path, 10)
        # failed launch attempts should re-queue the URL
        assert scout._summaries
    assert ffpuppet.return_value.launch.call_count == 2
    assert ffpuppet.return_value.clean_up.call_count == 2


@mark.parametrize(
    "urls, input_data",
    [
        # no urls to process
        ([], {}),
        # load urls
        (["http://a.c/", "http://b.a.c/d"], {"a.c": {"*": ["/"], "b": ["/d"]}}),
    ],
)
def test_site_scout_load_dict(urls, input_data):
    """test SiteScout.load_dict()"""
    with SiteScout(None) as scout:
        assert not scout._active
        scout.load_dict(input_data)
        for url in scout._urls:
            assert str(url) in urls
        assert len(urls) == len(scout._urls)


@mark.parametrize(
    "url, result",
    [
        # domain and tld (missing scheme)
        ("a.b", "http://a.b/"),
        # subdomain, domain and tld
        ("a.b.c", "http://a.b.c/"),
        # with scheme https
        ("https://a.b.c/", "https://a.b.c/"),
        # with scheme http
        ("http://a.b.c", "http://a.b.c/"),
        # with port
        ("a.b:1234", "http://a.b:1234/"),
        # port and path
        ("a.b:1234/c", "http://a.b:1234/c"),
        # port, path, parameters, query and fragment
        ("a.b/c;p?q=1&q2#f", "http://a.b/c;p?q=1&q2#f"),
        # normalizing domain and scheme
        ("HTTPS://A.B.C/eFg1", "https://a.b.c/eFg1"),
    ],
)
def test_site_scout_load_str(url, result):
    """test SiteScout.load_str() success"""
    with SiteScout(None) as scout:
        scout.load_str(url)
        assert scout._urls
        assert result in str(scout._urls[0])


@mark.parametrize(
    "url",
    [
        # invalid scheme
        "ftp://a.b.c",
        # unparsable domain
        "http://..a.c/",
        # unsupported subdomain
        f"http://{NO_SUBDOMAIN}.a.c/",
    ],
)
def test_site_scout_load_str_failures(url):
    """test SiteScout.load_str() failures"""
    with SiteScout(None) as scout:
        assert scout.load_str(url) is None


def test_site_scout_load_collision():
    """test loading an existing URL"""
    existing = "a.b.c"
    with SiteScout(None) as scout:
        scout.load_str(existing)
        assert len(scout._urls) == 1
        scout.load_str(existing)
        assert len(scout._urls) == 1
        scout.load_dict({"b.c": {"a": ["/"]}})
        assert len(scout._urls) == 1


@mark.parametrize(
    "active, jobs, completed, target, results, not_found, avg_dur, force",
    [
        # nothing running
        (0, 1, 0, 0, 0, 0, 0, False),
        # running with single site to visit
        (1, 1, 0, 1, 0, 0, 0, False),
        # running with single site to visit, forced report
        (1, 1, 0, 1, 0, 0, 0, True),
        # typical scenario
        (2, 3, 4, 10, 1, 1, 33, False),
    ],
)
def test_site_scout_status(
    mocker,
    tmp_path,
    active,
    jobs,
    completed,
    target,
    results,
    not_found,
    avg_dur,
    force,
):
    """test Status()"""
    mocker.patch("site_scout.site_scout.perf_counter", side_effect=count(start=1))
    dst = tmp_path / "status.txt"
    status = Status(dst, rate_limit=2)
    assert status
    assert status._next == 0
    status.report(
        active, jobs, completed, target, results, not_found, avg_dur, force=force
    )
    assert dst.is_file()
    assert status._next > 0
    next_report = status._next
    status.report(
        active, jobs, completed, target, results, not_found, avg_dur, force=force
    )
    if not force:
        assert status._next == next_report
    else:
        assert status._next > next_report
    output = dst.read_text()
    if not_found:
        assert "Not Found :" in output
    else:
        assert "Not Found :" not in output
    if avg_dur:
        assert "Avg Duration :" in output
    else:
        assert "Avg Duration :" not in output


@mark.parametrize(
    "size, limit, randomize, visits",
    [
        # empty
        (0, 0, True, 1),
        # empty with limit
        (0, 1, True, 1),
        # no limit
        (10, 0, True, 1),
        # enforce limit
        (10, 2, True, 1),
        # no limit
        (10, 0, False, 1),
        # enforce limit
        (10, 2, False, 1),
        # visit list 2x
        (5, 0, False, 2),
        # limit and repeat list
        (10, 2, False, 2),
    ],
)
def test_site_scout_schedule_urls(size, limit, randomize, visits):
    """test Status.schedule_urls()"""
    with SiteScout(None) as scout:
        # prepare scout._urls
        scout._urls = list(range(size))
        scout.schedule_urls(url_limit=limit, randomize=randomize, visits=visits)
        if limit and size >= limit:
            assert len(scout._urls) == limit * visits
        else:
            assert len(scout._urls) == size * visits
        if not randomize:
            assert scout._urls[-1] == 0
            assert scout._urls[0] == (limit - 1) if limit else (size - 1)
        if visits > 1:
            if limit > 0:
                assert scout._urls[0] == scout._urls[limit]
            elif size > 0:
                assert scout._urls[0] == scout._urls[size]


def test_site_scout_skip_not_found():
    """test SiteScout._skip_not_found()"""
    with SiteScout(None) as scout:
        scout._urls = [URL("a"), URL("b"), URL("c")] * 3
        scout._skip_not_found("a")
        assert len(scout._urls) == 6
        assert all(x.domain in ("b", "c") for x in scout._urls)
        assert len(scout._summaries) == 3
        assert all(x.url.domain == "a" for x in scout._summaries)
        assert all(x.not_found for x in scout._summaries)


def test_site_scout_skip_remaining(mocker):
    """test SiteScout._skip_remaining()"""
    active = mocker.Mock(spec_set=Visit)
    with SiteScout(None) as scout:
        scout._active = [active]
        scout._urls = [mocker.Mock(spec_set=URL)]
        scout._skip_remaining()
        assert not scout._active
        assert not scout._urls
        assert active.cleanup.call_count == 1


@mark.parametrize(
    "data, msg",
    [
        # empty
        ({}, "No data found"),
        # not a dict
        ([], "Invalid data"),
        # valid
        ({"d": {"s": ["/"]}}, None),
        # empty domain name
        ({"": {"s": ["/"]}}, "Domain must be a string"),
        # empty domain entry
        ({"d": {}}, "Invalid domain entry: 'd'"),
        # empty subdomain name
        ({"d": {"": ["/"]}}, "Subdomain must be a string"),
        # empty subdomain entry
        ({"d": {"s": []}}, "Invalid subdomain entry: 's' in 'd'"),
        # empty path entry
        ({"d": {"s": [""]}}, "Path must be a string starting with '/'"),
    ],
)
def test_verify_dict(data, msg):
    """test verify_dict()"""
    assert verify_dict(data) == msg

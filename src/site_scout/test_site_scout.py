# type: ignore
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
# pylint: disable=protected-access
from itertools import chain, count, cycle

from ffpuppet import BrowserTerminatedError, LaunchError, Reason
from pytest import mark, raises

from .explorer import State
from .site_scout import _LOAD_AVG, SiteScout, Status, Visit, verify_dict
from .url import URL


@mark.parametrize("explore", (True, False))
@mark.parametrize("alias", (None, "foo"))
def test_visit_basic(mocker, explore, alias):
    """test Visit"""
    exp = mocker.patch("site_scout.site_scout.Explorer", autospec=True).return_value
    puppet = mocker.patch("site_scout.site_scout.FFPuppet", autospec=True).return_value
    url = URL("foo")
    if alias is not None:
        url.alias = alias
    visit = Visit(puppet, url, explorer=exp if explore else None)
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
    summary = visit.summary()
    if explore:
        assert exp.close.call_count == 1
        assert summary.explore_duration is not None
        assert summary.load_duration is not None
        assert summary.state is not None
        if alias is not None:
            assert summary.url_loaded is None
        else:
            assert summary.url_loaded is not None
    else:
        assert summary.explore_duration is None
        assert summary.load_duration is None
        assert summary.state is None
        assert summary.url_loaded is None
    if alias is None:
        assert summary.identifier == str(url)
    else:
        assert summary.identifier == alias
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
    explorer.return_value.status.state = State.CLOSED
    explorer.return_value.status.url_loaded = ""
    ffpuppet = mocker.patch("site_scout.site_scout.FFPuppet", autospec=True)
    ffpuppet.return_value.is_healthy.return_value = False
    ffpuppet.return_value.reason = reason
    ffpuppet.return_value.save_logs.side_effect = save_logs
    getenv = mocker.patch("site_scout.site_scout.getenv", autospec=True)
    getenv.return_value = "collection-name"
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
    "reason, explore, state",
    [
        (Reason.ALERT, True, State.EXPLORING),
        (Reason.EXITED, True, State.CLOSED),
        (Reason.CLOSED, True, State.LOAD_FAILURE),
        (Reason.CLOSED, True, State.NOT_FOUND),
        (Reason.CLOSED, True, State.UNHANDLED_ERROR),
        (Reason.EXITED, False, None),
    ],
)
def test_site_scout_process_complete_summaries(
    mocker, tmp_path, reason, explore, state
):
    """test SiteScout._process_complete() summaries"""

    # pylint: disable=unused-argument
    def save_logs(dst_path, logs_only=False):
        dst_path.mkdir(exist_ok=True)

    explorer = mocker.patch("site_scout.site_scout.Explorer", autospec=True)
    explorer.return_value.status.load_duration = 1.0
    explorer.return_value.status.explore_duration = 2.0
    explorer.return_value.status.state = state
    explorer.return_value.status.url_loaded = "foo"
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
        assert scout._summaries[0].identifier == "http://foo/"
        assert scout._summaries[0].force_closed == (reason == Reason.CLOSED)
        assert scout._summaries[0].has_result == (reason == Reason.ALERT)
        if explore:
            assert explorer.return_value.close.call_count == 1
            assert scout._summaries[0].state == state
            assert scout._summaries[0].load_duration == 1.0
            assert scout._summaries[0].explore_duration == 2.0
        else:
            assert scout._summaries[0].state is None
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
):
    """test SiteScout.run()"""

    # pylint: disable=unused-argument
    def save_logs(dst_path, logs_only=False):
        dst_path.mkdir(exist_ok=True)

    mocker.patch("site_scout.site_scout.dump", autospec=True)
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
    "urls, input_data, omit_urls",
    [
        # no urls to process
        ([], {}, False),
        # load urls
        (["http://a.c/", "http://b.a.c/d"], {"a.c": {"*": ["/"], "b": ["/d"]}}, False),
        #
        (["http://a.c/"], {"a.c": {"*": ["/"]}}, True),
    ],
)
def test_site_scout_load_dict(urls, input_data, omit_urls):
    """test SiteScout.load_dict()"""
    with SiteScout(None, omit_urls=omit_urls) as scout:
        assert not scout._active
        scout.load_dict(input_data)
        for url in scout._urls:
            assert str(url) in urls
        assert len(urls) == len(scout._urls)
        if omit_urls:
            assert all(x.alias == "REDACTED" for x in scout._urls)
        else:
            assert all(x.alias is None for x in scout._urls)


@mark.parametrize("alias", ["foo", None])
@mark.parametrize("omit_urls", [True, False])
def test_site_scout_load_str(alias, omit_urls):
    """test SiteScout.load_str() success"""
    with SiteScout(None, omit_urls=omit_urls) as scout:
        scout.load_str("http://a.com/", alias=alias)
        assert scout._urls
        assert str(scout._urls[0]) == "http://a.com/"
        if alias is not None:
            assert scout._urls[0].alias == "foo"
        elif omit_urls:
            assert scout._urls[0].alias == "REDACTED"
        else:
            assert scout._urls[0].alias is None


def test_site_scout_load_str_failure():
    """test SiteScout.load_str() failures"""
    with SiteScout(None) as scout:
        assert scout.load_str("foo") is None


def test_site_scout_load_collision():
    """test loading an existing URL"""
    existing = "a.b.com"
    with SiteScout(None) as scout:
        scout.load_str(existing)
        assert len(scout._urls) == 1
        scout.load_str(existing)
        assert len(scout._urls) == 1
        scout.load_dict({"b.com": {"a": ["/"]}})
        assert len(scout._urls) == 1


@mark.parametrize(
    "active, jobs, completed, target, results, not_found, load_failure, avg_dur,"
    "visit_rate, force",
    [
        # nothing running
        (0, 1, 0, 0, 0, 0, 0, 0, True, False),
        # running with single site to visit
        (1, 1, 0, 1, 0, 0, 0, 0, False, False),
        # running with single site to visit, forced report
        (1, 1, 0, 1, 0, 0, 0, 0, False, True),
        # typical scenario
        (2, 3, 4, 10, 1, 1, 1, 33, True, False),
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
    load_failure,
    avg_dur,
    visit_rate,
    force,
):
    """test Status()"""
    mocker.patch("site_scout.site_scout.perf_counter", side_effect=count(start=1))
    dst = tmp_path / "status.txt"
    status = Status(dst, rate_limit=2)
    assert status
    assert status._next == 0
    status.report(
        active,
        jobs,
        completed,
        target,
        results,
        not_found,
        load_failure,
        avg_duration=avg_dur,
        include_rate=visit_rate,
        force=force,
    )
    assert dst.is_file()
    assert status._next > 0
    next_report = status._next
    status.report(
        active,
        jobs,
        completed,
        target,
        results,
        not_found,
        load_failure,
        avg_dur,
        force=force,
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
    if load_failure:
        assert "Load Failures :" in output
    else:
        assert "Load Failures :" not in output
    if avg_dur:
        assert "Avg Duration :" in output
    else:
        assert "Avg Duration :" not in output
    if visit_rate:
        assert "Visit Rate :" in output
    else:
        assert "Visit Rate :" not in output
    if _LOAD_AVG:
        assert "System Load :" in output
    else:
        assert "System Load :" not in output


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


def test_site_scout_skip_url():
    """test SiteScout._skip_url()"""
    with SiteScout(None) as scout:
        scout._urls = [URL("a"), URL("b"), URL("c")] * 3
        scout._skip_url(URL("a"), state=State.NOT_FOUND)
        assert len(scout._urls) == 6
        assert all(x.domain in ("b", "c") for x in scout._urls)
        assert len(scout._summaries) == 3
        assert all(x.identifier == "http://a/" for x in scout._summaries)
        assert all(x.state == State.NOT_FOUND for x in scout._summaries)


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

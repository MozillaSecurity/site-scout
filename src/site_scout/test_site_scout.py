# type: ignore
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
# pylint: disable=protected-access
from itertools import chain, count, cycle

from pytest import mark

from .browser_wrapper import BrowserArgs, BrowserState, BrowserWrapper
from .explorer import Explorer, State
from .site_scout import _LOAD_AVG, SiteScout, Status, Visit
from .url import URL


@mark.parametrize("explore", ("all", None))
@mark.parametrize("alias", (None, "foo"))
def test_visit_basic(mocker, explore, alias):
    """test Visit"""
    exp = mocker.patch("site_scout.site_scout.Explorer", autospec=True).return_value
    browser = mocker.Mock(spec_set=BrowserWrapper)
    url = URL("foo")
    if alias is not None:
        url.alias = alias
    visit = Visit(browser, url, explorer=exp if explore else None)
    assert visit.is_active()
    assert visit.duration() != visit.duration()
    assert visit.browser == browser
    assert browser.close.call_count == 0
    assert browser.cleanup.call_count == 0
    if explore:
        assert visit.explorer == exp
        assert exp.close.call_count == 0
    else:
        assert visit.explorer is None
    visit.close()
    assert not visit.is_active()
    assert visit.duration() == visit.duration()
    assert browser.close.call_count == 1
    assert browser.cleanup.call_count == 0
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
    assert browser.close.call_count == 1
    assert browser.cleanup.call_count == 1
    if explore:
        assert exp.close.call_count == 1


@mark.parametrize("explore", [None, "all"])
def test_site_scout_launch(mocker, explore, tmp_path):
    """test SiteScout._launch()"""
    mocker.patch("site_scout.site_scout.Explorer", autospec=True)
    args = BrowserArgs(tmp_path / "bin", 10, 10)
    with SiteScout(
        mocker.Mock(spec_set=BrowserWrapper), args, explore=explore
    ) as scout:
        assert not scout._active
        assert scout._launch(URL("someurl"), 1)
        assert scout._active
        assert scout._active[0].is_active()
        assert scout._active[0].idle_timestamp is None
        assert scout._active[0].browser is not None
        assert str(scout._active[0].url) == "http://someurl/"
        if explore:
            assert scout._active[0].explorer is not None
        else:
            assert scout._active[0].explorer is None


def test_site_scout_launch_failure(mocker, tmp_path):
    """test SiteScout._launch() failed"""
    mocker.patch("site_scout.site_scout.Explorer", autospec=True)
    browser = mocker.Mock(spec_set=BrowserWrapper)
    browser.return_value.launch.return_value = False
    args = BrowserArgs(tmp_path / "bin", 10, 10)
    with SiteScout(browser, args) as scout:
        assert not scout._launch(URL("someurl"), 1)
        assert browser.return_value.launch.call_count == 1
        assert browser.return_value.cleanup.call_count == 1
        assert scout._launch_failures == 1
        assert not scout._active


def test_site_scout_close(mocker):
    """test SiteScout.close()"""
    active = mocker.Mock(spec_set=Visit)
    complete = mocker.Mock(spec_set=Visit)
    with SiteScout(mocker.Mock(spec_set=BrowserWrapper), None, explore="all") as scout:
        scout._active = [active]
        scout._complete = [complete]
        scout.close()
        assert not scout._active
        assert not scout._complete
    assert active.cleanup.call_count == 1
    assert complete.cleanup.call_count == 1


@mark.parametrize(
    "urls, is_healthy, timeout, active, explore",
    [
        # no urls to process
        ([], True, False, 0, None),
        # one active
        ([URL("foo")], True, False, 1, None),
        # multiple active
        ([URL("foo"), URL("bar")], True, False, 2, None),
        # one complete
        ([URL("foo")], False, False, 0, None),
        # timeout
        ([URL("foo")], True, True, 0, None),
        # explorer closes before browser
        ([URL("foo")], True, False, 0, "all"),
    ],
)
def test_site_scout_process_active(mocker, urls, is_healthy, timeout, active, explore):
    """test SiteScout._process_active()"""
    explorer = mocker.patch(
        "site_scout.site_scout.Explorer", autospec=True
    ).return_value
    explorer.is_running.return_value = False
    browser = mocker.Mock(spec_set=BrowserWrapper)
    browser.return_value.is_healthy.return_value = is_healthy
    browser.return_value.create_explorer.return_value = explorer
    with SiteScout(browser, None, coverage=True, explore=explore) as scout:
        assert not scout._active
        assert scout._coverage
        for url in urls:
            scout._launch(url, 1)
        total_active = len(scout._active)
        # setup state
        if timeout:
            for visit in scout._active:
                visit._start_time = 0
        # run and verify
        scout._process_active(30, idle_usage=0)
        assert len(scout._active) == active
        assert active or total_active == len(scout._complete)
        for entry in scout._active:
            assert entry.is_active()
            assert not entry.browser.close.call_count
        for entry in scout._complete:
            assert not entry.is_active()
            assert entry.browser.close.call_count
            if is_healthy and not explore:
                assert entry.browser.dump_coverage.call_count
            else:
                assert not entry.browser.dump_coverage.call_count


def test_site_scout_process_active_idle(mocker):
    """test SiteScout._process_active() idle"""
    mocker.patch("site_scout.site_scout.perf_counter", side_effect=count())
    with SiteScout(
        mocker.Mock(spec_set=BrowserWrapper), None, coverage=True, explore=None
    ) as scout:
        assert not scout._active
        scout._launch(URL("foo"), 1)
        assert len(scout._active) == 1
        assert not scout._complete
        scout._active[0]._start_time = 0
        scout._active[0].browser.is_idle.return_value = True
        # set idle
        assert scout._active[0].idle_timestamp is None
        scout._process_active(30, idle_usage=10, idle_wait=10, min_visit=1)
        assert scout._active
        assert scout._active[0].idle_timestamp is not None
        # reset idle
        scout._active[0].browser.is_idle.return_value = False
        scout._process_active(30, idle_usage=10, idle_wait=10, min_visit=1)
        assert scout._active
        assert scout._active[0].idle_timestamp is None
        # set idle
        scout._active[0].browser.is_idle.return_value = True
        scout._process_active(30, idle_usage=10, idle_wait=10, min_visit=1)
        assert scout._active
        assert scout._active[0].idle_timestamp is not None
        # idle
        scout._process_active(30, idle_usage=10, idle_wait=1, min_visit=1)
        assert not scout._active
        assert len(scout._complete) == 1
        assert scout._complete[0].idle_timestamp is not None


@mark.parametrize(
    "urls, statue, reports",
    [
        # no urls to process
        ([], None, 0),
        # interesting result
        ([URL("foo")], BrowserState.RESULT, 1),
        # multiple interesting results
        ([URL("foo"), URL("bar")], BrowserState.RESULT, 2),
        # uninteresting result
        ([URL("foo")], BrowserState.CLOSED, 0),
        # uninteresting result
        ([URL("foo")], BrowserState.EXITED, 0),
    ],
)
def test_site_scout_process_complete(mocker, tmp_path, urls, statue, reports):
    """test SiteScout._process_complete()"""

    # pylint: disable=unused-argument
    def save_report(dst_path, logs_only=False):
        dst_path.mkdir(exist_ok=True)

    explorer = mocker.Mock(spec_set=Explorer)
    explorer.status.state = State.CLOSED
    explorer.status.url_loaded = ""
    browser = mocker.Mock(spec_set=BrowserWrapper)
    browser.return_value.is_healthy.return_value = False
    browser.return_value.state.return_value = statue
    browser.return_value.save_report.side_effect = save_report
    browser.return_value.create_explorer.return_value = explorer
    getenv = mocker.patch("site_scout.site_scout.getenv", autospec=True)
    getenv.return_value = "collection-name"
    prefs = tmp_path / "prefs.js"
    prefs.touch()
    report_dst = tmp_path / "reports"
    report_dst.mkdir()
    args = BrowserArgs(tmp_path / "bin", 10, 10, prefs_file=prefs)
    with SiteScout(browser, args, explore="all") as scout:
        assert not scout._active
        for url in urls:
            scout._launch(url, 1)
        assert scout._active or not urls
        scout._process_active(30)
        assert not scout._active
        assert len(scout._complete) == len(urls)
        assert not scout._summaries
        assert scout._process_complete(report_dst) == reports
        assert len(scout._summaries) == len(urls)
    assert sum(1 for _ in report_dst.iterdir()) == reports
    assert explorer.close.call_count == len(urls)


@mark.parametrize(
    "browser_state, explore, explorer_state",
    [
        (BrowserState.RESULT, "all", State.EXPLORING),
        (BrowserState.EXITED, "all", State.CLOSED),
        (BrowserState.CLOSED, "all", State.LOAD_FAILURE),
        (BrowserState.CLOSED, "all", State.NOT_FOUND),
        (BrowserState.CLOSED, "all", State.UNHANDLED_ERROR),
        (BrowserState.EXITED, None, None),
    ],
)
def test_site_scout_process_complete_summaries(
    mocker, tmp_path, browser_state, explore, explorer_state
):
    """test SiteScout._process_complete() summaries"""

    # pylint: disable=unused-argument
    def save_report(dst_path, logs_only=False):
        dst_path.mkdir(exist_ok=True)

    explorer = mocker.Mock(spec_set=Explorer)
    explorer.status.load_duration = 1.0
    explorer.status.explore_duration = 2.0
    explorer.status.state = explorer_state
    explorer.status.url_loaded = "foo"
    browser = mocker.Mock(spec_set=BrowserWrapper)
    browser.return_value.is_healthy.return_value = False
    browser.return_value.state.return_value = browser_state
    browser.return_value.save_report.side_effect = save_report
    browser.return_value.create_explorer.return_value = explorer

    with SiteScout(browser, None, explore=explore) as scout:
        assert not scout._active
        scout._launch(URL("foo"), 1)
        assert len(scout._active) == 1
        scout._process_active(30)
        assert not scout._active
        assert len(scout._complete) == 1
        assert not scout._summaries
        scout._process_complete(tmp_path)
        assert len(scout._summaries) == 1
        assert scout._summaries[0].duration > 0
        assert scout._summaries[0].identifier == "http://foo/"
        assert scout._summaries[0].force_closed == (
            browser_state == BrowserState.CLOSED
        )
        assert scout._summaries[0].has_result == (browser_state == BrowserState.RESULT)
        if explore is not None:
            assert explorer.close.call_count == 1
            assert scout._summaries[0].state == explorer_state
            assert scout._summaries[0].load_duration == 1.0
            assert scout._summaries[0].explore_duration == 2.0
        else:
            assert scout._summaries[0].state is None
            assert scout._summaries[0].load_duration is None
            assert scout._summaries[0].explore_duration is None


@mark.parametrize(
    "urls, state, jobs, reports, use_fm, status, explore, result_limit, runtime_limit",
    [
        # no urls to process
        ([], None, 1, 0, False, True, None, 0, 0),
        # interesting result
        ([URL("foo")], BrowserState.RESULT, 1, 1, False, False, None, 0, 0),
        # job > work
        ([URL("foo")], BrowserState.RESULT, 2, 1, False, False, None, 0, 0),
        # multiple interesting results
        (
            [URL("foo"), URL("bar")],
            BrowserState.RESULT,
            1,
            2,
            False,
            False,
            None,
            0,
            0,
        ),
        # work > jobs
        (
            [URL("1"), URL("2"), URL("3"), URL("4")],
            BrowserState.RESULT,
            2,
            4,
            False,
            False,
            None,
            0,
            0,
        ),
        # uninteresting result
        ([URL("foo")], BrowserState.CLOSED, 1, 0, False, False, None, 0, 0),
        # domain rate limit
        (
            [URL("foo"), URL("foo")],
            BrowserState.CLOSED,
            1,
            0,
            False,
            False,
            None,
            0,
            0,
        ),
        # timeout
        ([URL("foo")], None, 1, 0, False, False, None, 0, 0),
        # report via FuzzManager
        ([URL("foo")], BrowserState.RESULT, 1, 1, True, False, None, 0, 0),
        # report via FuzzManager (explore)
        ([URL("foo")], BrowserState.RESULT, 1, 1, True, False, "all", 0, 0),
        # report status
        ([URL("foo")], BrowserState.RESULT, 1, 1, False, True, None, 0, 0),
        # hit result limit
        (
            [URL("foo"), URL("bar")],
            BrowserState.RESULT,
            1,
            1,
            False,
            False,
            None,
            1,
            0,
        ),
        # hit runtime limit
        ([URL("foo"), URL("bar")], None, 1, 0, False, False, None, 0, 1),
        # explore
        ([URL("foo"), URL("bar")], None, 1, 0, False, False, "all", 0, 0),
    ],
)
def test_site_scout_run(
    mocker,
    tmp_path,
    urls,
    state,
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
    def save_report(dst_path, logs_only=False):
        dst_path.mkdir(exist_ok=True)

    mocker.patch("site_scout.site_scout.dump", autospec=True)
    mocker.patch("site_scout.site_scout.sleep", autospec=True)
    mocker.patch("site_scout.site_scout.perf_counter", side_effect=count())
    browser = mocker.Mock(spec_set=BrowserWrapper)
    reporter = mocker.patch("site_scout.site_scout.FuzzManagerReporter", autospec=True)
    reporter.return_value.submit.return_value = (1337, "[@ sig]")
    if state:
        # only first pass is running
        browser.return_value.is_healthy.side_effect = chain([True], cycle([False]))
    browser.return_value.state.return_value = state
    browser.return_value.save_report.side_effect = save_report

    report_dst = tmp_path / "reports"
    report_dst.mkdir()
    args = BrowserArgs(tmp_path / "bin", 10, 10)
    with SiteScout(browser, args, explore=explore, fuzzmanager=use_fm) as scout:
        assert not scout._active
        scout._urls = urls
        scout.run(
            report_dst,
            10,
            instance_limit=jobs,
            status_report=tmp_path / "status.txt" if status else None,
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
    browser = mocker.Mock(spec_set=BrowserWrapper)
    # one launch failure and one successful launch
    browser.return_value.launch.side_effect = (False, True)
    with SiteScout(browser, None, explore=False) as scout:
        assert not scout._active
        scout._urls = [URL("test")]
        scout.run(tmp_path, 10)
        # failed launch attempts should re-queue the URL
        assert scout._summaries
    assert browser.return_value.launch.call_count == 2
    assert browser.return_value.cleanup.call_count == 2


@mark.parametrize(
    "urls, input_data, omit_urls",
    [
        # no urls to process
        ([], {}, False),
        # load urls
        (["http://a.c/", "http://b.a.c/d"], {"a.c": {"": ["/"], "b": ["/d"]}}, False),
        #
        (["http://a.c/"], {"a.c": {"": ["/"]}}, True),
    ],
)
def test_site_scout_load_db(mocker, urls, input_data, omit_urls):
    """test SiteScout.load_db()"""
    with SiteScout(
        mocker.Mock(spec_set=BrowserWrapper), None, omit_urls=omit_urls
    ) as scout:
        assert not scout._active
        scout.load_db(input_data)
        for url in scout._urls:
            assert str(url) in urls
        assert len(urls) == len(scout._urls)
        if omit_urls:
            assert all(x.alias == "REDACTED" for x in scout._urls)
        else:
            assert all(x.alias is None for x in scout._urls)


@mark.parametrize("alias", ["foo", "", None])
@mark.parametrize("omit_urls", [True, False])
def test_site_scout_load_str(mocker, alias, omit_urls):
    """test SiteScout.load_str() success"""
    with SiteScout(
        mocker.Mock(spec_set=BrowserWrapper), None, omit_urls=omit_urls
    ) as scout:
        scout.load_str("http://a.com/", alias=alias)
        assert scout._urls
        assert str(scout._urls[0]) == "http://a.com/"
        if alias:
            assert scout._urls[0].alias == "foo"
        elif omit_urls:
            assert scout._urls[0].alias == "REDACTED"
        else:
            assert scout._urls[0].alias is None


def test_site_scout_load_str_failure(mocker):
    """test SiteScout.load_str() failures"""
    with SiteScout(mocker.Mock(spec_set=BrowserWrapper), None) as scout:
        assert scout.load_str("foo") is None


def test_site_scout_load_collision(mocker):
    """test loading an existing URL"""
    existing = "a.b.com"
    with SiteScout(mocker.Mock(spec_set=BrowserWrapper), None) as scout:
        scout.load_str(existing)
        assert len(scout._urls) == 1
        scout.load_str(existing)
        assert len(scout._urls) == 1
        scout.load_db({"b.com": {"a": ["/"]}})
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
def test_site_scout_schedule_urls(mocker, size, limit, randomize, visits):
    """test Status.schedule_urls()"""
    with SiteScout(mocker.Mock(spec_set=BrowserWrapper), None) as scout:
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


def test_site_scout_skip_url(mocker):
    """test SiteScout._skip_url()"""
    with SiteScout(mocker.Mock(spec_set=BrowserWrapper), None) as scout:
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
    with SiteScout(mocker.Mock(spec_set=BrowserWrapper), None) as scout:
        scout._active = [active]
        scout._urls = [mocker.Mock(spec_set=URL)]
        scout._skip_remaining()
        assert not scout._active
        assert not scout._urls
        assert active.cleanup.call_count == 1

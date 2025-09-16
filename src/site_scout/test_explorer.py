# type: ignore
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
# pylint: disable=protected-access
from page_explorer import PageLoad
from pytest import mark, raises

from .explorer import Explorer, ExplorerError, ExplorerMode, State


@mark.parametrize(
    "get_value, explore_return, state, title",
    (
        # navigation failed
        (ExplorerError(), False, State.LOADING, None),
        # failed to get page title
        (PageLoad.SUCCESS, True, State.CLOSED, None),
        # server not found
        (PageLoad.FAILURE, False, State.NOT_FOUND, "Server Not Found"),
        # problem loading page
        (PageLoad.FAILURE, False, State.LOAD_FAILURE, "Problem loading page"),
        # unknown navigate/load failure
        (PageLoad.FAILURE, False, State.UNHANDLED_ERROR, "error title"),
        # browser crashed
        (PageLoad.SUCCESS, False, State.EXPLORING, "title"),
        # successful interaction and browser closed
        (PageLoad.SUCCESS, True, State.CLOSED, "title"),
        # get timeout
        (PageLoad.TIMEOUT, True, State.CLOSED, "title"),
    ),
)
def test_explorer(mocker, tmp_path, get_value, explore_return, state, title):
    """test Explorer()"""
    page_explorer = mocker.patch(
        "site_scout.explorer.PageExplorer", autospec=True
    ).return_value.__enter__.return_value
    page_explorer.current_url = "http://foo.foo"
    page_explorer.explore.return_value = explore_return
    page_explorer.get.side_effect = (get_value,)
    page_explorer.is_connected.return_value = False
    page_explorer.title = title
    with Explorer(tmp_path, 0, "http://foo.foo", pause=0) as explorer:
        assert not explorer._can_skip.is_set()
        # allow explore thread to complete
        explorer._thread.join(timeout=10)
        explorer.close()
        assert not explorer.is_running()
        assert explorer._can_skip.is_set()
        assert explorer.status.state == state
        if get_value in (PageLoad.SUCCESS, PageLoad.TIMEOUT):
            assert explorer.status.load_duration > 0
            assert explorer.status.url_loaded == "http://foo.foo"
        else:
            assert explorer.status.load_duration is None
            assert explorer.status.url_loaded is None
        if explore_return:
            assert explorer.status.explore_duration > 0
        else:
            assert explorer.status.explore_duration is None


def test_explorer_failed_create_page_explorer(mocker, tmp_path):
    """test Explorer() failed to create PageExplorer"""
    mocker.patch("site_scout.explorer.PageExplorer", side_effect=ExplorerError("test"))
    with Explorer(tmp_path, 0, "http://foo.foo") as explorer:
        with raises(RuntimeError, match=r"Explorer\.close\(\) not called"):
            assert explorer.status is None
        explorer.close()
        assert explorer.status.state == State.INITIALIZING
        assert explorer.status.load_duration is None
        assert explorer.status.explore_duration is None


def test_explorer_failed_init(mocker, tmp_path):
    """test Explorer() failed to create PageExplorer"""
    mocker.patch("site_scout.explorer.PageExplorer", autospec=True)
    fake_event = mocker.patch("site_scout.explorer.Event", autospec=True)
    fake_event.return_value.wait.return_value = False
    with (
        raises(RuntimeError, match="PageExplorer thread did not unblock"),
        Explorer(tmp_path, 0, "http://a.foo") as _,
    ):
        pass


@mark.parametrize("mode", (ExplorerMode.ALL, ExplorerMode.LOAD))
def test_explorer_modes(mocker, tmp_path, mode):
    """test Explorer()"""
    page_explorer = mocker.patch(
        "site_scout.explorer.PageExplorer", autospec=True
    ).return_value.__enter__.return_value
    page_explorer.current_url = "http://foo.foo"
    page_explorer.explore.return_value = State.CLOSED
    page_explorer.get.return_value = PageLoad.SUCCESS
    page_explorer.is_connected.return_value = False
    page_explorer.title = "title"
    with Explorer(tmp_path, 0, "http://foo.foo", mode=mode, pause=0) as explorer:
        assert not explorer._can_skip.is_set()
        # allow explore thread to complete
        explorer._thread.join(timeout=10)
        explorer.close()
        if mode == ExplorerMode.ALL:
            assert page_explorer.explore.call_count == 1
        elif mode == ExplorerMode.LOAD:
            assert page_explorer.explore.call_count == 0

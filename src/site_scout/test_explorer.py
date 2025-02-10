# type: ignore
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
# pylint: disable=protected-access
from pytest import mark, raises

from .explorer import Explorer, ExplorerError, State


@mark.parametrize(
    "state, get_return, explore_return",
    (
        (State.NOT_FOUND, False, False),
        (State.EXPLORING, True, False),
        (State.CLOSED, True, True),
    ),
)
def test_explorer(mocker, tmp_path, state, get_return, explore_return):
    """test Explorer()"""
    page_explorer = mocker.patch(
        "site_scout.explorer.PageExplorer", autospec=True
    ).return_value.__enter__.return_value
    page_explorer.current_url = "http://foo.foo"
    page_explorer.explore.return_value = explore_return
    page_explorer.get.return_value = get_return
    with Explorer(tmp_path, 0, "http://foo.foo") as explorer:
        assert not explorer._can_skip.is_set()
        # allow explore thread to complete
        explorer._thread.join(timeout=10)
        explorer.close()
        assert not explorer.is_running()
        assert explorer._can_skip.is_set()
        assert explorer.state() == state.name
        if state == State.NOT_FOUND:
            assert explorer.not_found()
            assert explorer.url_loaded is None
        else:
            assert not explorer.not_found()
            assert explorer.url_loaded == "http://foo.foo"
        if get_return:
            assert explorer.get_duration() > 0
        else:
            assert explorer.get_duration() is None
        if explore_return:
            assert explorer.explore_duration() > 0
        else:
            assert explorer.explore_duration() is None


def test_explorer_failed_create_page_explorer(mocker, tmp_path):
    """test Explorer() failed to create PageExplorer"""
    mocker.patch("site_scout.explorer.PageExplorer", side_effect=ExplorerError("test"))
    with Explorer(tmp_path, 0, "http://foo.foo") as explorer:
        # allow explore thread to complete
        explorer._thread.join(timeout=10)
        assert explorer.state() == State.CONNECTING.name
        assert explorer.get_duration() is None
        assert explorer.explore_duration() is None


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

# type: ignore
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
# pylint: disable=protected-access
from pytest import mark

from .explorer import Explorer, State


@mark.parametrize(
    "state, get_return, explore_return",
    [
        (State.LOADING, False, False),
        (State.EXPLORING, True, False),
        (State.CLOSED, True, True),
    ],
)
def test_explorer(mocker, tmp_path, state, get_return, explore_return):
    """test Explorer()"""
    page_explorer = mocker.patch(
        "site_scout.explorer.PageExplorer", autospec=True
    ).return_value.__enter__.return_value
    page_explorer.get.return_value = get_return
    page_explorer.explore.return_value = explore_return
    with Explorer(tmp_path, 0, "http://foo.foo") as explorer:
        assert not explorer._can_skip.is_set()
        explorer.close()
        assert not explorer.is_running()
        assert explorer._can_skip.is_set()
        assert explorer.state() == state.name
        if get_return:
            assert explorer._status.get_duration > 0
        else:
            assert explorer._status.get_duration is None
        if explore_return:
            assert explorer._status.explore_duration > 0
        else:
            assert explorer._status.explore_duration is None

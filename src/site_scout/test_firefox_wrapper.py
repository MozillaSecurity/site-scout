# type: ignore
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
# pylint: disable=protected-access
from ffpuppet import BrowserTerminatedError, BrowserTimeoutError, Reason
from pytest import raises

from .browser_wrapper import BrowserArgs, BrowserState
from .explorer import ExplorerMode
from .firefox_wrapper import FirefoxWrapper
from .url import URL


def test_firefox_wrapper_basic(mocker, tmp_path):
    """test FirefoxWrapper basic"""
    args = BrowserArgs(tmp_path / "firefox", 10, 10)
    assert FirefoxWrapper.environment_manager(0, args) is None
    ffp = mocker.patch(
        "site_scout.firefox_wrapper.FFPuppet", autospec=True
    ).return_value
    with FirefoxWrapper(args, tmp_path, None) as browser:
        working_path = browser._working_path
        assert working_path.is_dir()
        assert not browser.debugger()
        # set is_healthy
        ffp.is_healthy.return_value = False
        assert not browser.is_healthy()
        assert ffp.is_healthy.call_count == 1
        # dump coverage should not run when is_healthy is false
        browser.dump_coverage()
        assert ffp.dump_coverage.call_count == 0
        # is_idle should always be true when is_healthy is false
        assert browser.is_idle(idle_limit=1)
        assert ffp.cpu_usage.call_count == 0
        # launch
        browser.launch(URL("a.com"), False, None, False)
        assert ffp.launch.call_count == 1
        assert ffp.clean_up.call_count == 0
        ffp.is_healthy.return_value = True
        # wait wrapper test
        browser.wait(0)
        assert ffp.wait.call_count == 1
        # dump coverage should run when is_healthy is true
        browser.dump_coverage()
        assert ffp.dump_coverage.call_count == 1
        # is_idle should check cpu usage when is_healthy is true
        ffp.cpu_usage.return_value = ((123, 99),)
        assert not browser.is_idle(idle_limit=10)
        assert ffp.cpu_usage.call_count == 1
        # close
        browser.close()
        assert ffp.close.call_count == 1
    assert ffp.clean_up.call_count == 1
    assert not working_path.is_dir()


def test_firefox_wrapper_state(mocker, tmp_path):
    """test FirefoxWrapper.state"""
    ffp = mocker.patch(
        "site_scout.firefox_wrapper.FFPuppet", autospec=True
    ).return_value
    args = BrowserArgs(tmp_path / "firefox", 10, 10)
    with FirefoxWrapper(args, tmp_path, None) as browser:
        ffp.reason = Reason.ALERT
        assert browser.state() == BrowserState.RESULT
        ffp.reason = Reason.WORKER
        assert browser.state() == BrowserState.RESULT
        ffp.reason = Reason.CLOSED
        assert browser.state() == BrowserState.CLOSED
        ffp.reason = Reason.EXITED
        assert browser.state() == BrowserState.EXITED


def test_firefox_wrapper_launch_failure(mocker, tmp_path):
    """test FirefoxWrapper.launch() failure"""
    ffp = mocker.patch(
        "site_scout.firefox_wrapper.FFPuppet", autospec=True
    ).return_value
    args = BrowserArgs(tmp_path / "firefox", 10, 10)
    with FirefoxWrapper(args, tmp_path, None) as browser:
        ffp.launch.side_effect = BrowserTimeoutError()
        with raises(BrowserTimeoutError):
            assert not browser.launch(
                "http://a.com", True, tmp_path, raise_on_failure=True
            )
        assert ffp.save_logs.call_count == 0
        assert ffp.clean_up.call_count == 1

        ffp.reset_mock()
        ffp.launch.side_effect = BrowserTerminatedError()
        assert not browser.launch(
            "http://a.com", True, tmp_path, raise_on_failure=False
        )
        assert ffp.save_logs.call_count == 0
        assert ffp.clean_up.call_count == 1

        ffp.reset_mock()
        ffp.launch.side_effect = BrowserTerminatedError()
        with raises(BrowserTerminatedError):
            browser.launch("http://a.com", True, tmp_path, raise_on_failure=True)
        assert ffp.save_logs.call_count == 1
        assert ffp.clean_up.call_count == 1


def test_firefox_wrapper_save_report(mocker, tmp_path):
    """test FirefoxWrapper.save_report()"""
    ffp = mocker.patch(
        "site_scout.firefox_wrapper.FFPuppet", autospec=True
    ).return_value
    prefs = tmp_path / "original_prefs.js"
    prefs.write_text("foo")
    args = BrowserArgs(tmp_path / "firefox", 10, 10, prefs_file=prefs)
    with FirefoxWrapper(args, tmp_path, None) as browser:
        browser.save_report(tmp_path)
        assert ffp.save_logs.call_count == 1
    assert (tmp_path / "prefs.js").is_file()


def test_firefox_wrapper_create_explorer(mocker, tmp_path):
    """test FirefoxWrapper.create_explorer()"""
    explorer = mocker.patch("site_scout.firefox_wrapper.Explorer", autospec=True)
    ffp = mocker.patch("site_scout.firefox_wrapper.FFPuppet", autospec=True)
    ffp.return_value.marionette = 12345
    args = BrowserArgs(tmp_path / "firefox", 10, 10)
    with FirefoxWrapper(args, tmp_path, None) as browser:
        assert browser.create_explorer(URL("a.com"), 10, mode=ExplorerMode.ALL, pause=5)
    explorer.assert_called_with(
        args.binary,
        12345,
        "http://a.com/",
        mode=ExplorerMode.ALL,
        load_wait=10,
        pause=5,
    )

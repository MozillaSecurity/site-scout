# type: ignore
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
# pylint: disable=protected-access
from subprocess import TimeoutExpired

from fxpoppet import ADBLaunchError, ADBProcess, Reason
from fxpoppet.adb_session import ADBSessionError
from fxpoppet.emulator.android import AndroidEmulator, AndroidEmulatorError
from pytest import mark, raises

from .browser_wrapper import BrowserArgs, BrowserState
from .fenix_wrapper import EmulatorPool, FenixEnvironmentManager, FenixWrapper
from .url import URL


def test_fenix_wrapper_invalid_apk(mocker, tmp_path):
    """test FenixWrapper invalid APK"""
    session = mocker.patch("site_scout.fenix_wrapper.ADBSession", autospec=True)
    session.get_package_name.return_value = None
    with raises(RuntimeError, match=r"Could not find package name."):
        FenixWrapper(
            BrowserArgs(tmp_path / "target.apk", 10, 10),
            tmp_path,
            mocker.Mock(spec_set=FenixEnvironmentManager),
        )


def test_fenix_wrapper_basic(mocker, tmp_path):
    """test FenixWrapper basic"""
    prefs = tmp_path / "custom_prefs.js"
    prefs.touch()
    mocker.patch("site_scout.fenix_wrapper.ADBSession", autospec=True)
    proc = mocker.patch(
        "site_scout.fenix_wrapper.ADBProcess", autospec=True
    ).return_value
    with FenixWrapper(
        BrowserArgs(tmp_path / "target.apk", 10, 10, prefs_file=prefs),
        tmp_path,
        mocker.Mock(spec_set=FenixEnvironmentManager),
    ) as browser:
        assert not browser.debugger()
        # launch
        browser.launch(URL("a.com"), False, None, False)
        assert proc.launch.call_count == 1
        assert proc.cleanup.call_count == 0
        # is_healthy
        proc.is_healthy.return_value = True
        assert browser.is_healthy()
        assert proc.is_healthy.call_count == 1
        # save report
        browser.save_report(tmp_path)
        assert (tmp_path / "prefs.js").is_file()
        # wait wrapper test
        browser.wait(0)
        assert proc.wait.call_count == 1
        # is_idle should check cpu usage when is_healthy is true
        proc.cpu_usage.return_value = ((123, 99),)
        assert not browser.is_idle(idle_limit=10)
        assert proc.cpu_usage.call_count == 1
        proc.is_healthy.return_value = False
        assert browser.is_idle(idle_limit=10)
        # close
        browser.close()
        assert proc.close.call_count == 1
    assert proc.cleanup.call_count == 1


@mark.parametrize(
    "state, reason",
    [
        (BrowserState.EXITED, Reason.EXITED),
        (BrowserState.RESULT, Reason.ALERT),
        (BrowserState.CLOSED, Reason.CLOSED),
    ],
)
def test_fenix_wrapper_state(mocker, tmp_path, state, reason):
    """test FenixWrapper.state()"""
    mocker.patch("site_scout.fenix_wrapper.ADBSession", autospec=True)
    mocker.patch("site_scout.fenix_wrapper.FenixEnvironmentManager", autospec=True)
    args = BrowserArgs(tmp_path / "target.apk", 10, 10)
    env_mgr = FenixWrapper.environment_manager(1, args)
    with FenixWrapper(args, tmp_path, env_mgr) as browser:
        browser._proc = mocker.Mock(spec_set=ADBProcess)
        browser._proc.reason = reason
        assert browser.state() == state


def test_fenix_wrapper_launch_failures(mocker, tmp_path):
    """test FenixWrapper.launch() failures"""
    session_cls = mocker.patch("site_scout.fenix_wrapper.ADBSession", autospec=True)
    proc_cls = mocker.patch("site_scout.fenix_wrapper.ADBProcess", autospec=True)
    env_mgr = mocker.Mock(spec_set=FenixEnvironmentManager)
    with FenixWrapper(
        BrowserArgs(tmp_path / "target.apk", 10, 10), tmp_path, env_mgr
    ) as browser:
        # no device
        env_mgr.select_device.return_value = None
        assert not browser.launch(URL("a.com"), False, None, False)
        # failed to create session
        env_mgr.reset_mock(return_value=True)
        session_cls.connect.side_effect = ADBSessionError("boot failed")
        assert not browser.launch(URL("a.com"), False, None, False)
        assert proc_cls.call_count == 0
        session_cls.reset_mock(side_effect=True)
        # failed to create proc
        proc_cls.side_effect = ADBLaunchError("launch-fail-test")
        assert not browser.launch(URL("a.com"), False, None, False)
        assert browser._session is not None
        assert browser._proc is None
        assert proc_cls.call_count == 1
        proc_cls.reset_mock(side_effect=True)
        # failed to launch (raise false)
        proc_cls.return_value.launch.side_effect = ADBLaunchError("launch-fail-test")
        browser._session = None
        assert not browser.launch(URL("a.com"), False, None, False)
        assert browser._session is not None
        assert browser._proc is not None
        # failed to launch (raise true)
        browser._session = None
        browser._proc = None
        with raises(ADBLaunchError, match="launch-fail-test"):
            browser.launch(URL("a.com"), False, None, True)


def test_emulator_pool_launch_and_prep_emulator(mocker, tmp_path):
    """test EmulatorPool launch and prepare emulator"""
    emu_cls = mocker.patch("site_scout.fenix_wrapper.AndroidEmulator", autospec=True)
    emu_cls.search_free_ports.return_value = 1234
    session_cls = mocker.patch("site_scout.fenix_wrapper.ADBSession", autospec=True)
    pool = EmulatorPool(1)
    # handle a launch failure
    emu_cls.side_effect = RuntimeError("testing-failed-launch")
    with raises(RuntimeError, match="testing-failed-launch"):
        pool._launch_emulator()
    assert emu_cls.create_avd.call_count == 1
    assert emu_cls.remove_avd.call_count == 1
    # handle process shutdown
    session_cls.connect.side_effect = ADBSessionError("boot failed")
    emu_cls.reset_mock(side_effect=True)
    emu = pool._launch_emulator()
    assert emu_cls.create_avd.call_count == 1
    assert emu_cls.remove_avd.call_count == 0
    assert not pool._prepare_emulator(emu, tmp_path / "fake.apk")
    assert session_cls.connect.call_count == 1
    assert session_cls.connect.return_value.device.shell.call_count == 0
    # success
    session_cls.reset_mock(side_effect=True)
    assert pool._prepare_emulator(emu, tmp_path / "fake.apk")
    assert session_cls.connect.call_count == 1
    assert session_cls.connect.return_value.device.shell.call_count == 7
    assert session_cls.connect.return_value.install.call_count == 1
    # prep failure
    session_cls.reset_mock()
    session_cls.connect.return_value.install.side_effect = ADBSessionError(
        "install failed"
    )
    assert not pool._prepare_emulator(emu, tmp_path / "fake.apk")
    assert session_cls.connect.call_count == 1


def test_emulator_pool_basic(mocker, tmp_path):
    """test EmulatorPool basic"""
    pool = EmulatorPool(1)
    assert pool._size_limit == 1
    assert not pool._in_use
    # launch failure
    mocker.patch.object(
        EmulatorPool, "_launch_emulator", side_effect=AndroidEmulatorError()
    )
    for _ in range(EmulatorPool.LAUNCH_FAILURE_LIMIT - 1):
        pool.manage_emulators(tmp_path / "target.apk")
    with raises(AndroidEmulatorError):
        pool.manage_emulators(tmp_path / "target.apk")
    # prepare fails
    emu = mocker.Mock(spec_set=AndroidEmulator)
    mocker.patch.object(EmulatorPool, "_launch_emulator", return_value=emu)
    mocker.patch.object(EmulatorPool, "_prepare_emulator", return_value=False)
    emu.poll.return_value = None
    pool.manage_emulators(tmp_path / "target.apk")
    assert emu.terminate.call_count == 1
    assert not pool._emulators
    # prepare success
    mocker.patch.object(EmulatorPool, "_prepare_emulator", return_value=True)
    emu.reset_mock(return_value=True)
    emu.port = 1234
    emu.poll.return_value = None
    pool.manage_emulators(tmp_path / "target.apk")
    assert pool._emulators
    assert "emulator-1234" in pool._emulators
    # select
    device = pool.select()
    assert device == "emulator-1234"
    assert "emulator-1234" in pool._in_use
    # release
    pool.release("emulator-1234")
    assert not pool._in_use
    # select - emulators not running
    pool._emulators["emulator-1234"].poll.return_value = 0
    assert pool.select() is None
    assert not pool._in_use
    # shutdown
    pool._emulators["emulator-1234"].poll.return_value = None
    emu.reset_mock()
    pool.shutdown("emulator-1234")
    assert emu.terminate.call_count == 1
    assert emu.wait.call_count == 1
    assert emu.cleanup.call_count == 1
    # cleanup
    emu.reset_mock()
    pool.cleanup()
    assert emu.terminate.call_count == 1
    assert emu.wait.call_count == 1
    assert emu.cleanup.call_count == 1
    # cleanup - failed to terminate
    emu.reset_mock()
    pool._emulators["emulator-1234"].poll.return_value = None
    pool._emulators["emulator-1234"].wait.side_effect = TimeoutExpired("foo", 1)
    pool.cleanup()
    assert emu.terminate.call_count == 1
    assert emu.wait.call_count == 1
    assert emu.emu.kill.call_count == 1


def test_emulator_pool_check_emulators(mocker):
    """test EmulatorPool._check_emulators()"""
    session_cls = mocker.patch("site_scout.fenix_wrapper.ADBSession", autospec=True)
    pool = EmulatorPool(1)
    # empty pool
    pool._check_emulators()
    # emulator not running
    emu = mocker.Mock(spec_set=AndroidEmulator)
    emu.poll.return_value = 0
    pool._emulators["test-1234"] = emu
    pool._check_emulators()
    assert session_cls.connect.call_count == 0
    # running emulator
    emu = mocker.Mock(spec_set=AndroidEmulator)
    emu.poll.return_value = None
    pool._emulators["test-1234"] = emu
    pool._check_emulators()
    assert session_cls.connect.call_count == 1
    assert emu.cleanup.call_count == 0
    # running emulator - connection failed
    session_cls.connect.side_effect = ADBSessionError("boot failed")
    pool._check_emulators()
    assert emu.cleanup.call_count == 1


def test_emulator_pool_trim_emulators(mocker):
    """test EmulatorPool._trim_emulators()"""
    session_cls = mocker.patch("site_scout.fenix_wrapper.ADBSession", autospec=True)
    pool = EmulatorPool(1)
    # empty pool
    pool._trim_emulators()
    # emulator running
    emu = mocker.Mock(spec_set=AndroidEmulator)
    emu.poll.return_value = None
    pool._emulators["test-1234"] = emu
    pool._trim_emulators()
    assert pool._emulators
    # emulator not running
    emu = mocker.Mock(spec_set=AndroidEmulator)
    emu.poll.return_value = 0
    pool._emulators["test-1234"] = emu
    pool._in_use.add("test-1234")
    pool._trim_emulators()
    assert not pool._emulators
    assert not pool._in_use
    assert session_cls.call_count == 0


def test_emulator_pool_shutdown_emulator(mocker):
    """test EmulatorPool._shutdown_emulator()"""
    # successful termination
    emu = mocker.Mock(spec_set=AndroidEmulator)
    emu.poll.return_value = None
    EmulatorPool._shutdown_emulator(emu)
    assert emu.terminate.call_count == 1
    assert emu.wait.call_count == 1
    assert emu.emu.kill.call_count == 0
    assert emu.cleanup.call_count == 1
    # failed termination
    emu.reset_mock()
    emu.wait.side_effect = TimeoutExpired("foo", 1)
    EmulatorPool._shutdown_emulator(emu)
    assert emu.terminate.call_count == 1
    assert emu.wait.call_count == 2
    assert emu.emu.kill.call_count == 1
    assert emu.cleanup.call_count == 1


def test_fenix_environment_manager_basic(mocker, tmp_path):
    """test FenixEnvironmentManager basic"""
    pool_cls = mocker.patch("site_scout.fenix_wrapper.EmulatorPool", autospec=True)
    pool_cls.return_value.select.return_value = "test-1234"
    env_mgr = FenixEnvironmentManager(1, BrowserArgs(tmp_path / "a.apk", 1, 1))
    assert env_mgr.select_device(tmp_path / "foo.apk") == "test-1234"
    env_mgr.release_device("test-1234")
    assert pool_cls.return_value.release.call_count == 1
    env_mgr.shutdown_device("test-1234")
    assert pool_cls.return_value.shutdown.call_count == 1
    env_mgr.cleanup()
    assert pool_cls.return_value.cleanup.call_count == 1

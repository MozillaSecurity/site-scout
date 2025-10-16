# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from __future__ import annotations

from contextlib import suppress
from logging import getLogger
from os import getenv
from shutil import copyfile
from time import perf_counter
from typing import TYPE_CHECKING

from fxpoppet import ADBLaunchError, ADBProcess, ADBSession, Reason
from fxpoppet.adb_session import ADBCommunicationError, ADBSessionError
from fxpoppet.emulator.android import AndroidEmulator, AndroidEmulatorError

from .browser_wrapper import (
    BrowserArgs,
    BrowserState,
    BrowserWrapper,
    EnvironmentManager,
)

if TYPE_CHECKING:
    from pathlib import Path

    from .explorer import Explorer, ExplorerMode
    from .url import URL


LOG = getLogger(__name__)


class FenixWrapper(BrowserWrapper):
    """FenixWrapper is used to add support for Firefox for Android."""

    __slots__ = ("_device", "_env_mgr", "_package", "_proc", "_session")

    def __init__(
        self, args: BrowserArgs, working_path: Path, env_mgr: FenixEnvironmentManager
    ) -> None:
        super().__init__(args, working_path, env_mgr)
        self._env_mgr = env_mgr
        self._package = ADBSession.get_package_name(args.binary)
        if self._package is None:
            LOG.error("FenixWrapper failed to find package name!")
            raise RuntimeError("Could not find package name.")
        self._proc: ADBProcess | None = None
        self._session: ADBSession | None = None
        # TODO: generate prefs if needed
        self._device: str | None = None

    def cleanup(self) -> None:
        if self._proc is not None:
            self._proc.cleanup()
        if self._session is not None:
            if self._session.connected:
                self._session.reverse_remove()
            self._session.disconnect()
        if self._device is not None:
            self._env_mgr.release_device(self._device)

    def close(self, force_close: bool = False) -> None:
        if self._proc is not None:
            self._proc.close()

    def create_explorer(
        self, url: URL, load_wait: int, mode: ExplorerMode, pause: int = 0
    ) -> Explorer:
        raise NotImplementedError()

    def debugger(self) -> bool:
        return False

    def dump_coverage(self, timeout: int = 15) -> None:
        raise NotImplementedError()

    @staticmethod
    def environment_manager(
        instance_limit: int, browser_args: BrowserArgs
    ) -> FenixEnvironmentManager:
        return FenixEnvironmentManager(instance_limit, browser_args)

    def is_healthy(self) -> bool:
        assert self._proc is not None
        return self._proc.is_healthy()

    def is_idle(self, idle_limit: int) -> bool:
        assert self._proc is not None
        if self._proc.is_healthy():
            # check that no processes are over the limit
            return not any(pct > idle_limit for _, pct in self._proc.cpu_usage())
        # not running
        return True

    def launch(
        self,
        url: str,
        explore: bool,
        log_path: Path | None,
        raise_on_failure: bool,
    ) -> bool:
        assert not explore, "explore not supported"
        assert self._package is not None
        assert self._proc is None
        assert self._session is None

        self._device = self._env_mgr.select_device(self.args.binary)
        if self._device is None:
            LOG.error("Failed to select target device!")
            return False

        LOG.debug("opening adb session...")
        session = ADBSession(self._device)
        with suppress(ADBCommunicationError, ADBSessionError):
            # emulator is expected to be running
            session.connect(as_root=True, boot_timeout=10)
        if not session.connected:
            LOG.error("FenixWrapper failed to connect to device!")
            return False
        LOG.debug("connected to device (%s)", session.device_id)
        self._session = session
        self._session.symbols[self._package] = self.args.binary.parent / "symbols"

        try:
            self._proc = ADBProcess(self._package, self._session)
        except ADBSessionError:
            LOG.error("FenixWrapper ADBProcess init failed!")
            return False

        try:
            self._proc.launch(
                str(url),
                env_mod={
                    # prevent crash reporter from touching the dmp files
                    "MOZ_CRASHREPORTER": "1",
                    "MOZ_CRASHREPORTER_NO_REPORT": "1",
                    "MOZ_CRASHREPORTER_SHUTDOWN": "1",
                },
                launch_timeout=self.args.launch_timeout,
                prefs_js=self.args.prefs_file,
            )
        except ADBLaunchError as exc:
            LOG.error("ADBProcess LaunchError: %s", exc)
            self.close()
            if raise_on_failure:
                raise
            return False
        return True

    def save_report(self, dst: Path) -> None:
        assert self._proc is not None
        self._proc.save_logs(dst)
        # save prefs file
        if self.args.prefs_file is not None:
            copyfile(self.args.prefs_file, dst / "prefs.js")

    def state(self) -> BrowserState:
        assert self._proc is not None
        assert self._proc.reason is not None
        if self._proc.reason == Reason.EXITED:
            return BrowserState.EXITED
        if self._proc.reason == Reason.ALERT:
            return BrowserState.RESULT
        return BrowserState.CLOSED

    def wait(self, timeout: float) -> bool:
        assert self._proc is not None
        return self._proc.wait(timeout=timeout)


class EmulatorPool:
    """Create and manage a collection of Android emulators.

    Attributes:
        _display_mode: Display mode of emulators.
        _emulators: Emulator objects.
        _in_use: Corresponding serials of emulators currently in use.
        _size_limit: Maximum number of emulators allowed.
    """

    __slot__ = ("_display_mode", "_emulators", "_in_use", "_size_limit")

    def __init__(self, size_limit: int, display_mode: str = "default") -> None:
        assert display_mode in {"default", "headless", "xvfb"}
        assert size_limit > 0
        self._display_mode = display_mode
        self._emulators: dict[str, AndroidEmulator] = {}
        self._in_use: set[str] = set()
        self._size_limit = size_limit

    def cleanup(self) -> None:
        """Terminate all running emulators.

        Args:
            None

        Returns:
            None
        """
        for emu in self._emulators.values():
            if emu.poll() is None:
                emu.terminate()
        deadline = perf_counter() + 60
        for emu in self._emulators.values():
            max_delay = max(deadline - perf_counter(), 1)
            # this can raise subprocess.TimeoutExpired
            emu.wait(timeout=max_delay)
            emu.cleanup()

    def manage_emulators(self, apk: Path) -> None:
        """Manage the emulator pool.

        Args:
            apk: Browser APK file.

        Returns:
            None
        """
        # check emulators are in a good state
        self._check_emulators()
        # remove emulators that are not running from the pool
        self._trim_emulators()
        # launch emulator if needed
        available = len(self._emulators) - len(self._in_use)
        assert available >= 0
        LOG.debug(
            "emulators: running %d, available %d, limit %d",
            len(self._emulators),
            available,
            self._size_limit,
        )
        if available == 0 and len(self._emulators) < self._size_limit:
            for attempts_remaining in reversed(range(2)):
                try:
                    emu = self._launch_emulator()
                except AndroidEmulatorError:
                    LOG.warning("Emulator launch attempt failed!")
                    if attempts_remaining > 0:
                        continue
                    raise
                break
            if self._prepare_emulator(emu, apk):
                serial = f"emulator-{emu.port}"
                assert serial not in self._emulators
                self._emulators[serial] = emu
            else:
                # emulator prep failed
                if emu.poll() is None:
                    emu.terminate()
                emu.wait(timeout=30)
                emu.cleanup()
                LOG.warning("Failed to prepare device")
        assert len(self._emulators) <= self._size_limit

    def release(self, serial: str) -> None:
        """Mark an emulator as not in use.

        Args:
            serial: Emulator to mark as not in use.

        Returns:
            None
        """
        assert serial in self._in_use
        self._in_use.remove(serial)

    def select(self) -> str | None:
        """Select available emulator.

        Args:
            None.

        Returns:
            None
        """
        available = set(self._emulators) - self._in_use
        assert available
        while available:
            selected = available.pop()
            # check if emulator is running
            if self._emulators[selected].poll() is None:
                self._in_use.add(selected)
                return selected
        LOG.debug("no running emulator available")
        return None

    def _check_emulators(self) -> None:
        for serial, emu in self._emulators.items():
            if emu.poll() is not None:
                # emulator is not running
                continue
            session = ADBSession(serial)
            with suppress(ADBCommunicationError, ADBSessionError):
                session.connect(as_root=False, boot_timeout=10)
            if not session.connected:
                LOG.warning("Cannot connect to device (%s)", serial)
                emu.terminate()
                emu.wait(timeout=30)
                emu.cleanup()

    def _launch_emulator(self) -> AndroidEmulator:
        port = AndroidEmulator.search_free_ports()
        avd_name = f"x86.{port:d}"
        LOG.debug("launching emulator (emulator-%d)...", port)
        AndroidEmulator.create_avd(avd_name)
        try:
            emu = AndroidEmulator(
                port=port,
                avd_name=avd_name,
                boot_timeout=300,
                headless=self._display_mode == "headless",
                xvfb=self._display_mode == "xvfb",
                emulator_output=getenv("EMULATOR_OUTPUT", "") == "1",
            )
        except:
            AndroidEmulator.remove_avd(avd_name)
            raise
        return emu

    def _prepare_emulator(self, emu: AndroidEmulator, apk: Path) -> bool:
        LOG.debug("preparing device (emulator-%d)...", emu.port)
        session = ADBSession(f"emulator-{emu.port}")
        with suppress(ADBCommunicationError, ADBSessionError):
            session.connect(as_root=True, boot_timeout=10)
        if not session.connected:
            LOG.warning("Cannot connect to prepare emulator")
            return False
        try:
            # disable some UI animations
            session.shell(["settings", "put", "global", "animator_duration_scale", "0"])
            session.shell(
                ["settings", "put", "global", "transition_animation_scale", "0"]
            )
            session.shell(["settings", "put", "global", "window_animation_scale", "0"])
            # prevent device throttling
            session.shell(["settings", "put", "global", "device_idle_enabled", "0"])
            session.shell(["settings", "put", "global", "low_power", "0"])
            session.shell(
                ["settings", "put", "global", "background_process_limit", "0"]
            )
            session.shell(["dumpsys", "deviceidle", "disable"])
            session.install(apk)
        except ADBCommunicationError:
            LOG.debug("ADBCommunicationError during prep")
            return False
        finally:
            session.disconnect()
        return True

    def _trim_emulators(self) -> None:
        # remove emulators that are not running
        not_running: list[str] = []
        for serial, emu in self._emulators.items():
            if emu.poll() is not None:
                not_running.append(serial)
        for serial in not_running:
            LOG.debug("removing emulator (%s) from pool", serial)
            if serial in self._in_use:
                LOG.warning("Emulator (%s) not running and in use!", serial)
                self._in_use.remove(serial)
            self._emulators.pop(serial)


class FenixEnvironmentManager(EnvironmentManager):
    """FenixEnvironmentManager is used to add support for Firefox for Android."""

    def __init__(self, instance_limit: int, browser_args: BrowserArgs) -> None:
        assert instance_limit > 0
        super().__init__()
        self._pool = EmulatorPool(
            instance_limit, display_mode=browser_args.display_mode or "default"
        )

    def select_device(self, apk: Path) -> str | None:
        """Select available device.

        Args:
            apk: Browser APK file.

        Returns:
            Device serial if a device is available otherwise None.
        """
        self._pool.manage_emulators(apk)
        return self._pool.select()

    def release_device(self, serial: str) -> None:
        """Release device.

        Args:
            serial: Device to mark as available.

        Returns:
            None
        """
        self._pool.release(serial)

    def cleanup(self) -> None:
        self._pool.cleanup()

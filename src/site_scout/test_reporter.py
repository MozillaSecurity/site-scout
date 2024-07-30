# type: ignore
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
from pytest import mark, raises

from .reporter import FuzzManagerReporter


@mark.parametrize(
    "include_cfg, aux_log",
    [
        # missing aux log
        (True, None),
        # missing binary .fuzzmanagerconf files
        (False, None),
        # Sanitizer logs
        (True, "log_ffp_asan_16365.log.18532.txt"),
        # minidump logs
        (True, "log_minidump_00.txt"),
    ],
)
def test_fuzzmanager_reporter_01(tmp_path, mocker, include_cfg, aux_log):
    """test FuzzManagerReporter"""
    collector = mocker.patch("site_scout.reporter.Collector", autospec=True)
    empty = tmp_path / "empty.bin"
    empty.touch()
    fm_config = None
    if include_cfg:
        fm_config = tmp_path / f"{empty.name}.fuzzmanagerconf"
        fm_config.write_text(
            "[Main]\n"
            "platform = x86-64\n"
            "product = mozilla-central\n"
            "product_version = 20230629-e784085dfb50\n"
            "os = linux"
        )
    result = tmp_path / "result"
    result.mkdir()
    (result / "log_stderr.txt").write_text("foo")
    (result / "log_stdout.txt").write_text("foo")
    if aux_log:
        (result / aux_log).write_text("foo")
    (result / "url.txt").write_text("foo")
    reporter = FuzzManagerReporter(empty, fm_config=fm_config)
    assert reporter.submit(result, {"duration": "1.0", "url": "foo"})
    assert collector.return_value.submit.call_count == 1


def test_fuzzmanager_reporter_02(tmp_path):
    """test FuzzManagerReporter missing fuzzmanagerconf"""

    with raises(FileNotFoundError):
        FuzzManagerReporter(tmp_path, fm_config=tmp_path / "missing")

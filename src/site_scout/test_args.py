# type: ignore
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
from platform import system

from pytest import mark, raises

from .args import TIME_LIMIT_DEBUG, TIME_LIMIT_DEFAULT, TIME_LIMIT_EXPLORE, parse_args


def test_parse_args_basic(tmp_path):
    """test parse_args()"""
    dummy_file = tmp_path / "dummy_file"
    dummy_file.touch()
    assert parse_args([str(dummy_file), "-i", str(dummy_file)])


def test_parse_args_checks(capsys, tmp_path):
    """test parse_args() checks"""
    dummy_file = tmp_path / "dummy_file"
    dummy_file.touch()

    with raises(SystemExit):
        parse_args(["missing", "-i", str(dummy_file)])
    assert "binary does not exist: 'missing'" in capsys.readouterr()[-1]

    with raises(SystemExit):
        parse_args([str(dummy_file), "-i", "missing"])
    assert "-i/--input does not exist: 'missing'" in capsys.readouterr()[-1]

    with raises(SystemExit):
        parse_args([str(dummy_file), "-i", str(dummy_file), "-o", "missing"])
    assert "-o/--output-path does not exist: 'missing'" in capsys.readouterr()[-1]

    with raises(SystemExit):
        parse_args([str(dummy_file), "-i", str(dummy_file), "--prefs", "missing"])
    assert "--prefs does not exist: 'missing'" in capsys.readouterr()[-1]

    with raises(SystemExit):
        parse_args([str(dummy_file), "-i", str(dummy_file), "--jobs", "0"])
    assert "--jobs must be >= 1" in capsys.readouterr()[-1]

    with raises(SystemExit):
        parse_args([str(dummy_file), "-i", str(dummy_file), "--log-limit", "-1"])
    assert "--log-limit must be >= 0" in capsys.readouterr()[-1]

    with raises(SystemExit):
        parse_args([str(dummy_file), "-i", str(dummy_file), "--memory-limit", "-1"])
    assert "--memory-limit must be >= 0" in capsys.readouterr()[-1]

    with raises(SystemExit):
        parse_args([str(dummy_file), "-i", str(dummy_file), "--time-limit", "-1"])
    assert "--time-limit must be > 0" in capsys.readouterr()[-1]


def test_parse_args_debugger(mocker, capsys, tmp_path):
    """test parse_args() rr/pernosco checks"""
    mocker.patch("site_scout.args.system", autospec=True, return_value="Linux")
    fake_which = mocker.patch("site_scout.args.which", autospec=True)
    dummy_file = tmp_path / "dummy_file"
    dummy_file.touch()

    with raises(SystemExit):
        parse_args([str(dummy_file), "-u", "foo", "--rr", "--fuzzmanager"])
    assert "rr not supported with --fuzzmanager" in capsys.readouterr()[-1]

    fake_which.return_value = None
    with raises(SystemExit):
        parse_args([str(dummy_file), "-u", "foo", "--rr"])
    assert "rr is not installed" in capsys.readouterr()[-1]

    fake_which.return_value = "rr"
    mocker.patch("site_scout.args.Path.read_bytes", autospec=True, return_value=b"99")
    with raises(SystemExit):
        parse_args([str(dummy_file), "-u", "foo", "--rr"])
    assert "perf_event_paranoid <= 1, but it is 99" in capsys.readouterr()[-1]


@mark.skipif(system() != "Linux", reason="Only supported on Linux")
def test_parse_args_coverage(capsys, mocker, tmp_path):
    """test parse_args() - coverage"""
    dummy_file = tmp_path / "dummy_file"
    dummy_file.touch()
    fake_getenv = mocker.patch("site_scout.args.getenv", autospec=True)

    assert parse_args([str(dummy_file), "-i", str(dummy_file), "--coverage"])

    fake_getenv.side_effect = ("", "", "")
    with raises(SystemExit):
        assert parse_args([str(dummy_file), "-i", str(dummy_file), "--coverage"])
    assert (
        "error: GCOV_PREFIX_STRIP must be set to use --coverage"
        in capsys.readouterr()[-1]
    )

    fake_getenv.side_effect = ("", "", "/", "")
    with raises(SystemExit):
        assert parse_args([str(dummy_file), "-i", str(dummy_file), "--coverage"])
    assert "error: GCOV_PREFIX must be set to use --coverage" in capsys.readouterr()[-1]

    fake_getenv.side_effect = ("", "", "/", "/")
    with raises(SystemExit):
        assert parse_args(
            [str(dummy_file), "-i", str(dummy_file), "--coverage", "--jobs", "2"]
        )
    assert (
        "error: Parallel jobs not supported with --coverage" in capsys.readouterr()[-1]
    )


@mark.parametrize(
    "extra_args, expected_time",
    [
        ([], TIME_LIMIT_DEFAULT),
        (["--explore"], TIME_LIMIT_EXPLORE),
        (["--pernosco"], TIME_LIMIT_DEBUG),
        (["--time-limit", "111"], 111),
    ],
)
def test_args_time_limit(mocker, tmp_path, extra_args, expected_time):
    """test parse_args() set time limit"""
    mocker.patch("site_scout.args.Path.read_bytes", autospec=True, return_value=b"0")
    mocker.patch("site_scout.args.system", autospec=True, return_value="Linux")
    mocker.patch("site_scout.args.which", autospec=True)

    dummy_file = tmp_path / "dummy_file"
    dummy_file.touch()
    args = parse_args([str(dummy_file), "-i", str(dummy_file), *extra_args])
    assert args.time_limit == expected_time

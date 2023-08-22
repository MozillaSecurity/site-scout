# type: ignore
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
from pytest import raises

from .args import parse_args


def test_args_01(tmp_path):
    """test parse_args()"""
    dummy_file = tmp_path / "dummy_file"
    dummy_file.touch()
    assert parse_args([str(dummy_file), "-i", str(dummy_file)])


def test_args_02(capsys, tmp_path):
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

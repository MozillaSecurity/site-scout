# type: ignore
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
from logging import NOTSET, disable

from ffpuppet import BrowserTerminatedError
from pytest import mark, raises

from .main import generate_prefs, load_jsonl, load_yml, main, scan_input
from .url_db import UrlDBError


@mark.parametrize(
    "args",
    [
        # only required args (use yml file)
        [],
        # log-level
        ["--log-level", "DEBUG"],
        # url
        ["-u", "http://mozilla.com"],
        # disable logging
        ["--disable-logging", "-u", "http://private-url.com"],
    ],
)
def test_main_01(caplog, capsys, mocker, tmp_path, args):
    """test main()"""
    browser = mocker.patch("site_scout.main.FirefoxWrapper", autospec=True)
    browser.return_value.is_healthy.return_value = False
    fake_bin = tmp_path / "fake_browser_bin"
    fake_bin.touch()
    default_args = [str(fake_bin), "-b", "firefox"]
    # -i or -u is required (create yml file)
    if "-u" not in args:
        # test yml loading path
        yml_file = tmp_path / "data.yml"
        yml_file.write_text("{'d':{'s':['/']}}")
        # test jsonl loading path
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text('{"a": "b"}\n')
        default_args.extend(["-i", str(yml_file), str(jsonl_file)])
    try:
        assert main(default_args + args) == 0
    finally:
        # always re-enable logging
        disable(level=NOTSET)
    assert browser.return_value.launch.call_count == 1
    if "--disable-logging" in args:
        # verify we are not sending any unexpected data to the console
        console = capsys.readouterr()
        assert not console.err
        assert not console.out
        assert "Running with logging disabled..." in caplog.text
        assert "Starting Site Scout..." not in caplog.text
        assert "Done." not in caplog.text
        assert "private-url.com" not in caplog.text
        assert len(caplog.text.splitlines()) == 1
    else:
        assert "Starting Site Scout..." in caplog.text
        assert "Done." in caplog.text
        assert "Running with logging disabled..." not in caplog.text
        assert len(caplog.text.splitlines()) > 1


@mark.parametrize(
    "yml_data",
    [
        # invalid yml data
        "\0",
        # invalid data in yml
        "{'d':[]}",
    ],
)
def test_main_load_yml_invalid(caplog, mocker, tmp_path, yml_data):
    """test main()"""
    browser = mocker.patch("site_scout.main.FirefoxWrapper", autospec=True)
    browser.return_value.is_healthy.return_value = False
    fake_bin = tmp_path / "fake_browser_bin"
    fake_bin.touch()
    yml_file = tmp_path / "data.yml"
    yml_file.write_text(yml_data)
    assert main([str(fake_bin), "-b", "firefox", "-i", str(yml_file)]) == 0
    assert "Starting Site Scout..." in caplog.text
    assert "Done." in caplog.text
    assert "Running with logging disabled..." not in caplog.text
    assert len(caplog.text.splitlines()) > 1


@mark.parametrize(
    "exception, exit_code",
    [
        # user abort
        (KeyboardInterrupt(), 0),
        # browser failed to launch
        (BrowserTerminatedError(), 1),
    ],
)
def test_main_02(mocker, tmp_path, exception, exit_code):
    """test main() exceptions"""
    mocker.patch("site_scout.main.SiteScout", autospec=True, side_effect=exception)
    fake_bin = tmp_path / "fake_browser_bin"
    fake_bin.touch()
    assert main([str(fake_bin), "-u", "a.b.c"]) == exit_code


def test_generate_prefs_01(tmp_path):
    """test generate_prefs()"""
    assert generate_prefs(dst=tmp_path).is_file()


def test_load_yml_simple(tmp_path):
    """test load_yml() with valid data"""
    (tmp_path / "sites.yml").write_text("{'d':{'s':['/']}}")
    assert len(load_yml(tmp_path / "sites.yml")) == 1


def test_load_yml_invalid_data(tmp_path):
    """test load_yml() with invalid data"""
    (tmp_path / "test.yml").write_text("\0")
    with raises(UrlDBError, match="Invalid yml"):
        load_yml(tmp_path / "test.yml")


def test_load_jsonl_simple(tmp_path):
    """test load_jsonl() with valid data"""
    (tmp_path / "sites.jsonl").write_text(
        '{"a.com": "423-456-7853"}\n'
        '{"b.com": "333-422-2222"}\n'
        '{"c.com": "123-456-7890"}\n'
        '{"d.com": ""}\n'
        '{"e.com": null}\n'
    )
    results = tuple(load_jsonl(tmp_path / "sites.jsonl"))
    assert len(results) == 5


@mark.parametrize(
    "data",
    [
        # empty file
        b"",
        # invalid chars ' should be "
        b"{'url': 'alias'}\n",
        # empty entry
        b"{}\n",
        # empty lines
        b"\n\n\n",
        # invalid data
        b'{"a": []}\n',
        # invalid data
        b'{"a"}\n',
        # invalid data
        b"[1]\n",
    ],
)
def test_load_jsonl_invalid_data(tmp_path, data):
    """test load_jsonl() with invalid data"""
    (tmp_path / "test.yml").write_bytes(data)
    assert not any(load_jsonl(tmp_path / "test.yml"))


def test_scan_input(tmp_path):
    """test scan_input()"""
    # no paths
    assert not any(scan_input([], ".foo"))
    # empty directory
    assert not any(scan_input([tmp_path], ".foo"))
    # single file
    (tmp_path / "test.yml").touch()
    assert any(scan_input([(tmp_path / "test.yml")], ".yml"))
    # single file in directory
    assert len(tuple(scan_input([tmp_path], ".yml"))) == 1
    # no suffix match
    assert not any(scan_input([tmp_path], ".foo"))
    # multiple files in directory
    (tmp_path / "test.jsonl").touch()
    results = tuple(scan_input([tmp_path], ".yml"))
    assert len(results) == 1
    assert results[0].suffix == ".yml"

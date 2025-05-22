# type: ignore
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
from logging import NOTSET, disable

from ffpuppet import BrowserTerminatedError
from pytest import mark

from .main import generate_prefs, load_input, main


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
    ffp = mocker.patch("site_scout.site_scout.FFPuppet", autospec=True)
    ffp.return_value.is_healthy.return_value = False
    fake_bin = tmp_path / "fake_browser_bin"
    fake_bin.touch()
    default_args = [str(fake_bin)]
    # -i or -u is required (create yml file)
    if "-u" not in args:
        data = tmp_path / "data.yml"
        data.write_text("{'d':{'s':['/']}}")
        default_args.extend(["-i", str(data)])
    try:
        assert main(default_args + args) == 0
    finally:
        # always re-enable logging
        disable(level=NOTSET)
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


def test_load_input_01(tmp_path):
    """test load_input()"""
    # empty list
    assert not tuple(load_input([]))
    # empty directory
    assert not tuple(load_input([tmp_path]))

    valid = "{'d':{'s':['/']}}"
    # single input file
    in_file = tmp_path / "sites.yml"
    in_file.write_text(valid)
    results = tuple(load_input([in_file]))
    assert len(results) == 1
    # multiple input files
    results = tuple(load_input([in_file, in_file]))
    assert len(results) == 2
    # single input directory
    results = tuple(load_input([tmp_path]))
    assert len(results) == 1

    # invalid file data
    (tmp_path / "ignore.txt").write_text("foo")
    (tmp_path / "bad.yml").write_text("{-")
    (tmp_path / "sites.yml").write_text(valid)
    (tmp_path / "a.yml").write_text("foo")
    (tmp_path / "cont-char.yml").write_bytes(b"\0")
    results = tuple(load_input([tmp_path]))
    assert len(results) == 1

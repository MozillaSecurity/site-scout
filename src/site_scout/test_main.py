# type: ignore
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
from pytest import mark

from .main import generate_prefs, main


@mark.parametrize(
    "args",
    [
        # only required args
        [],
        # log-level
        ["--log-level", "DEBUG"],
        # url
        ["-u", "http://mozilla.com"],
    ],
)
def test_main_01(mocker, tmp_path, args):
    """test main()"""
    ffp = mocker.patch("site_scout.site_scout.FFPuppet", autospec=True)
    ffp.return_value.is_healthy.return_value = False
    fake_bin = tmp_path / "fake_browser_bin"
    fake_bin.touch()
    default_args = [str(fake_bin)]
    # -i or -u is required
    if "-u" not in args:
        data = tmp_path / "data.yml"
        data.write_text("{}")
        default_args.extend(["-i", str(data)])
    main(default_args + args)


def test_generate_prefs_01(tmp_path):
    """test generate_prefs()"""
    assert generate_prefs(dst=tmp_path).is_file()

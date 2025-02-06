# type: ignore
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
# pylint: disable=missing-docstring,protected-access
from pytest import mark, raises

from .split_collection import main, parse_args, split_collection
from .url_collection import UrlCollection


def _generate_urls(entries):
    assert entries >= 0
    db = {}
    for entry in range(entries):
        domain = f"domain-{entry % 7}"
        subdomain = f"sub-{entry % 3}"
        if domain not in db:
            db[domain] = {}
        if subdomain not in db[domain]:
            db[domain][subdomain] = []
        db[domain][subdomain].append(f"/path-{entry}")
    urls = UrlCollection(db)
    assert len(urls) == entries
    return urls


@mark.parametrize("entries", [1001, 1, 0])
def test_split_collection_01(tmp_path, entries):
    """test split_collection()"""
    target_size = 500
    urls = _generate_urls(entries)
    # calculate minimum number of expected files
    src = tmp_path / "src.yml"
    urls.save_yml(src)
    expected_files = src.stat().st_size // target_size if entries else 0
    src.unlink()
    # split collection
    dst = tmp_path / "dst"
    dst.mkdir()
    file_count = split_collection(urls, dst, "split-", size=target_size)
    assert file_count >= expected_files
    files = tuple(dst.glob("split-*"))
    assert len(files) == file_count
    # verify split files contain all entries
    total = 0
    for file in files:
        total += len(UrlCollection.load_yml(file))
    assert total == entries


def test_parse_args_01(capsys, tmp_path):
    """test parse_args()"""
    # missing src
    with raises(SystemExit):
        parse_args(["missing.yml"])
    assert "error: missing.yml is not a file." in capsys.readouterr().err
    # missing dst
    file = tmp_path / "test.yml"
    file.touch()
    with raises(SystemExit):
        parse_args([str(file), "-o", "missing_dst"])
    assert "error: missing_dst is not a directory." in capsys.readouterr().err
    # success
    assert parse_args([str(file), "-o", str(tmp_path)])


def test_main_01(tmp_path):
    """test main()"""
    data_file = tmp_path / "test.yml"
    # empty data file
    data_file.write_text("{}")
    assert main([str(data_file), "-o", str(tmp_path)]) == 1
    # single entry
    data_file.write_text("{'a': {'b': ['/']}}")
    assert main([str(data_file), "-o", str(tmp_path)]) == 0

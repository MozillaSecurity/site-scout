# type: ignore
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
# pylint: disable=missing-docstring,protected-access
from pytest import mark, raises

from .url_collection import UrlCollection, main, parse_args

BASIC_URL_DB = {
    "d1": {"*": ["/", "/d1-path1"]},
    "d2": {
        "sd2-1": ["/sd2-path1"],
        "sd2-2": ["/"],
    },
}


@mark.parametrize(
    "url_str, expected",
    [
        ("http://domain.com/", "http://domain.com/"),
        ("http://sub.domain.com/", "http://sub.domain.com/"),
        ("https://sub.domain.com/", "http://sub.domain.com/"),
        ("sub.domain.com", "http://sub.domain.com/"),
        ("domain.com", "http://domain.com/"),
        ("domain.com/ ", "http://domain.com/"),
        ("domain.com/test ", "http://domain.com/test"),
        ("user:pass@domain.com", "http://domain.com/"),
        ("domain.co.uk", "http://domain.co.uk/"),
        ("domain.com/test", "http://domain.com/test"),
        ("domain.com/test?123", "http://domain.com/test?123"),
        ("domain.com/test#123", "http://domain.com/test#123"),
        ("HTTP://SUB.DOMAIN.COM/TEST", "http://sub.domain.com/TEST"),
        ("domain.com/El Niño", "http://domain.com/El%20Ni%C3%B1o"),
        ("fóò.café.fr/test", "http://xn--f-tgac.xn--caf-dma.fr/test"),
        ("*.domain.com", None),
        ("failed-parse", None),
    ],
)
def test_url_collection_parse_url(url_str, expected):
    """test UrlCollection._parse_url()"""
    parsed = UrlCollection._parse_url(url_str, default_subdomain=None)
    if expected is None:
        assert parsed is None
    else:
        assert parsed
        assert str(parsed) == expected


def test_url_collection_01():
    """test empty UrlCollection()"""
    urls = UrlCollection()
    assert urls is not None
    assert not urls
    assert len(urls) == 0
    assert not tuple(urls.domains)


def test_url_collection_02(tmp_path):
    """test simple UrlCollection()"""
    urls = UrlCollection(BASIC_URL_DB)
    assert urls
    assert len(urls) == 4
    assert len(tuple(urls.domains)) == 2
    # save and load
    yml = tmp_path / "urls.yml"
    urls.save_yml(yml)
    urls = UrlCollection.load_yml(yml)
    assert urls
    # add/remove urls
    assert len(urls) == 4
    assert urls.add_url("test.com")
    assert len(urls) == 5
    assert urls.remove_url("test.com")
    assert not urls.remove_url("test.com")
    # try adding a urls that can't be parsed
    assert not urls.unparsable
    assert not urls.add_url("invalid")
    assert "invalid" in urls.unparsable
    # add a list of urls from a file
    txt = tmp_path / "urls_list.txt"
    txt.write_text("1.test.com\n\n2.test.com\n#ignored.com\nother.com")
    assert urls.add_list(txt)
    assert not urls.add_list(txt)
    assert len(urls) == 7


def test_url_collection_sorting():
    """test UrlCollection() sorting"""
    urls = UrlCollection({"a.com": {"b": ["/z", "/", "/b"]}})
    assert urls._db["a.com"]["b"][0] == "/z"
    assert urls._db["a.com"]["b"][1] == "/"
    urls.sort_paths()
    assert urls._db["a.com"]["b"][0] == "/"
    assert urls._db["a.com"]["b"][1] == "/b"
    # added paths should be sorted
    urls.add_url("b.a.com/a")
    assert urls._db["a.com"]["b"][0] == "/"
    assert urls._db["a.com"]["b"][1] == "/a"
    assert urls._db["a.com"]["b"][2] == "/b"


def test_parse_args(capsys, tmp_path):
    """test parse_args()"""
    #
    data_file = tmp_path / "test.yml"
    args = parse_args([str(data_file)])
    assert args.url_db == data_file

    #
    with raises(SystemExit):
        parse_args([str(data_file), "-l", "missing_list.txt"])
    assert "error: missing_list.txt is not a file." in capsys.readouterr().err


def test_main(tmp_path):
    """test main()"""

    # no data file
    data_file = tmp_path / "test.yml"
    assert main([str(data_file)]) == 0
    assert not data_file.exists()

    # invalid data file
    data_file.write_bytes(b"\0")
    assert main([str(data_file)]) == 1
    data_file.unlink()

    # remove url (data file does not exists)
    assert main([str(data_file), "-r", "test.com"]) == 0
    assert not data_file.exists()

    # add single url
    assert main([str(data_file), "-u", "test.com", "-d"]) == 0
    assert data_file.exists()

    # remove url (data file now exists)
    file_size = data_file.stat().st_size
    assert main([str(data_file), "-r", "test.com"]) == 0
    assert data_file.stat().st_size < file_size
    assert data_file.exists()

    # add unparsable
    file_size = data_file.stat().st_size
    assert main([str(data_file), "-u", "blah"]) == 0
    assert data_file.stat().st_size == file_size

    # remove unparsable
    file_size = data_file.stat().st_size
    assert main([str(data_file), "-r", "blah"]) == 0
    assert data_file.stat().st_size == file_size

    # add urls from list file
    txt = tmp_path / "urls_list.txt"
    txt.write_text("1.test.com\n\n2.test.com\n#ignored.com\nother.com")
    file_size = data_file.stat().st_size
    assert main([str(data_file), "-l", str(txt)]) == 0
    assert data_file.stat().st_size > file_size

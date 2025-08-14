# type: ignore
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
# pylint: disable=protected-access

from pytest import mark, raises

from .url import URL, URLParseError


def test_url_str():
    """test URL.__str__()"""
    assert str(URL("a.c", subdomain="b", path="/d")) == "http://b.a.c/d"


@mark.parametrize(
    "domain, subdomain, path, scheme, expected",
    [
        # with a subdomain
        ("a.c", "b", "/d", "http", "http://b.a.c/d"),
        # without a subdomain
        ("a.c", None, "/d", "http", "http://a.c/d"),
        # more complex
        ("a.c:1337", "x.y", "/1/2/3.html", "https", "https://x.y.a.c:1337/1/2/3.html"),
        # normalize case (domain and subdomain only)
        ("aB.cD", "Ab", "/Ef", "HTTP", "http://ab.ab.cd/Ef"),
        # normalize case (missing subdomain)
        ("aB.cD", None, "/Ef", "HTTP", "http://ab.cd/Ef"),
        # idna encode domain
        ("é.c", "ò", "/", "http", "http://xn--jda.xn--9ca.c/"),
        # percent encode non ascii chars path
        ("a.c", None, "/ñ%41=1", "http", "http://a.c/%C3%B1%41=1"),
        # previously encoded path
        ("a.c", None, "/%C3%B1o", "http", "http://a.c/%C3%B1o"),
    ],
)
def test_url_create(domain, subdomain, path, scheme, expected):
    """test URL.create()"""
    url = URL.create(domain, subdomain=subdomain, path=path, scheme=scheme)
    assert str(url) == expected
    assert url.uid


@mark.parametrize(
    "domain, subdomain, path, scheme, msg",
    [
        # bad domain
        ("*", None, "/", "http", "Invalid domain '*'"),
        # bad domain
        ("", None, "/", "http", "Invalid domain ''"),
        # bad domain
        ("b a.d", None, "/", "http", "Invalid domain 'b a.d'"),
        # bad domain
        ("b_a.d", None, "/", "http", "Invalid domain 'b_a.d'"),
        # bad domain
        (".a.c", None, "/", "http", "Invalid domain '.a.c'"),
        # bad subdomain
        ("a.c", "$.foo", "/", "http", r"Invalid subdomain '\$.foo'"),
        # bad subdomain
        ("a.c", ".a", "/", "http", "Invalid subdomain '.a'"),
        # bad subdomain
        ("a.c", "..a", "/", "http", "Invalid subdomain '..a'"),
        # bad subdomain
        ("d.c", "b a", "/", "http", "Invalid subdomain 'b a'"),
        # bad subdomain
        ("d.c", "b:a", "/", "http", "Invalid subdomain 'b:a'"),
        # bad path
        ("a.c", None, "", "http", "Path must begin with '/'"),
        # bad path
        ("a.c", None, "foo", "http", "Path must begin with '/'"),
        # bad scheme
        ("a.c", None, "/", "foo", "Unsupported scheme 'foo'"),
    ],
)
def test_url_create_failures(domain, subdomain, path, scheme, msg):
    """test URL.create() - failures"""
    with raises(URLParseError, match=msg):
        URL.create(domain, subdomain=subdomain, path=path, scheme=scheme)


@mark.parametrize(
    "url_str, expected",
    [
        # http scheme, domain and tld
        ("http://domain.com/", "http://domain.com/"),
        # https scheme, domain and tld
        ("https://domain.com/", "https://domain.com/"),
        # http scheme, domain, subdomain and tld
        ("http://sub.domain.com/", "http://sub.domain.com/"),
        # https scheme, domain, subdomain and tld
        ("https://sub.domain.com/", "https://sub.domain.com/"),
        # subdomain, domain and tld (missing scheme)
        ("sub.domain.com", "http://sub.domain.com/"),
        # domain and tld (missing scheme)
        ("domain.com", "http://domain.com/"),
        # trailing space
        ("domain.com ", "http://domain.com/"),
        # trailing space after path
        ("domain.com/ ", "http://domain.com/"),
        # trailing space after path
        ("domain.com/test ", "http://domain.com/test"),
        # contains credentials
        ("user:pass@domain.com", "http://domain.com/"),
        # multipart tld
        ("domain.co.uk", "http://domain.co.uk/"),
        # with path
        ("domain.com/test", "http://domain.com/test"),
        # query
        ("domain.com/test?123", "http://domain.com/test?123"),
        # fragment
        ("domain.com/test#123", "http://domain.com/test#123"),
        # normalize domain, subdomain and scheme
        ("HTTP://SUB.DOMAIN.COM/TeSt", "http://sub.domain.com/TeSt"),
        # non-ascii and space in path
        ("domain.com/El Niño", "http://domain.com/El%20Ni%C3%B1o"),
        # non-ascii in domain and subdomain
        ("fóò.café.fr/test", "http://xn--f-tgac.xn--caf-dma.fr/test"),
        # with port
        ("a.com:1234", "http://a.com:1234/"),
        # port and path
        ("a.com:1234/c", "http://a.com:1234/c"),
        # port, path, parameters, query and fragment
        ("a.com:80/c;p?q=1&q2#f", "http://a.com:80/c;p?q=1&q2#f"),
    ],
)
def test_url_parse(url_str, expected):
    """test URL.parse()"""
    assert str(URL.parse(url_str)) == expected


@mark.parametrize(
    "url_str, msg",
    [
        ("", "Missing domain"),
        ("/", "Missing domain"),
        ("http://", "Missing domain"),
        ("user:name@", "Missing domain"),
        (".com", "Missing domain"),
        ("data:text/html,TEST", "Missing domain"),
        (".a.com", r"Invalid subdomain$"),
        ("test", "Missing suffix"),
        ("192.168.1.1", "Missing suffix"),
        ("about:config", "Missing suffix"),
        ("foo.notarealtld", "Missing suffix"),
        ("http://a.com:foo", "Port must be a number"),
        ("http://a.com:0", "Invalid port number"),
        ("http://[a.com/", "Invalid URL"),
    ],
)
def test_url_parse_invalid_strings(url_str, msg):
    """test URL.parse() - invalid strings()"""
    with raises(URLParseError, match=msg):
        URL.parse(url_str)

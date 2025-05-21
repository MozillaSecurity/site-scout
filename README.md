Site Scout
==========
[![CI](https://github.com/MozillaSecurity/site-scout/actions/workflows/ci.yml/badge.svg)](https://github.com/MozillaSecurity/site-scout/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/MozillaSecurity/site-scout/branch/main/graph/badge.svg)](https://codecov.io/gh/MozillaSecurity/site-scout)
[![Matrix](https://img.shields.io/badge/chat-%23fuzzing-green?logo=matrix)](https://matrix.to/#/#fuzzing:mozilla.org)
[![PyPI](https://img.shields.io/pypi/v/site-scout)](https://pypi.org/project/site-scout)

Site Scout is a tool that can identify and report issues that are triggered in the wild.
A URL or a collection of URLs must be provided. All results are collected and reported.

The primary goal is to find issues such as crashes, assertion failures and other
issues detected by AddressSanitizer, UndefinedBehaviourSanitizer and ThreadSanitizer.

Installation
------------

Install the latest version from PyPI:

```
python3 -m pip install site-scout --upgrade
```

Usage
-----

Visit one or more URLs (typically used to reproduce an issue):
```
site-scout <firefox-bin> -u <url> [<url> ...]
```

-or-

Visit a collection of URLs:
```
site-scout <firefox-bin> -i <urls>.yml [<urls>.yml ...]
```

Input YAML Layout
-----------------

URLs to visit are stored in the following format:
```yml
# subdomain.domain.tld/path
domain.tld:
  subdomain:
  - /path
# '*' is used to omit the subdomain
github.com:
  '*':
  - /MozillaSecurity/site-scout
mozilla.org:
  www:
  - /
  - /firefox/browsers/mobile/
  - /firefox/new/
```

# Site Scout

[![CI](https://github.com/MozillaSecurity/site-scout/actions/workflows/ci.yml/badge.svg)](https://github.com/MozillaSecurity/site-scout/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/MozillaSecurity/site-scout/branch/main/graph/badge.svg)](https://codecov.io/gh/MozillaSecurity/site-scout)
[![Matrix](https://img.shields.io/badge/chat-%23fuzzing-green?logo=matrix)](https://matrix.to/#/#fuzzing:mozilla.org)
[![PyPI](https://img.shields.io/pypi/v/site-scout)](https://pypi.org/project/site-scout)

Site Scout is a tool that can identify issues that are triggered by visiting live
content. The primary goal is to create a tool that can be used in automation to collect
debugging data for issues such as crashes and assertion failures. A collection of URLs
to visit must be provided.

## Installation

Install the latest version from PyPI:

```bash
pip install site-scout --upgrade
```

## Usage

Visit one or more URLs (typically used to reproduce an issue):
```bash
site-scout <firefox-bin> -u <url> [<url> ...]
```

Visit a collection of URLs from a file:
```bash
site-scout <firefox-bin> -i <urls>.yml [<urls>.yml ...]
```

## Input Formats

**YML** - URLs to visit are stored in the following format:
```yml
# subdomain.domain.tld/path
domain.tld:
  subdomain:
  - /path
# '' is used to omit the subdomain
github.com:
  '':
  - /MozillaSecurity/site-scout
mozilla.org:
  www:
  - /
  - /firefox/browsers/mobile/
  - /firefox/new/
```

**JSONL** - URLs to visits with optional alias:
```jsonl
{"url1.tld": "alias-string-1"}
{"url2.tld/path": "alias-string-2"}
{"sub.url3.tld": null}
```

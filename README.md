Site Scout
==========
[![Task Status](https://community-tc.services.mozilla.com/api/github/v1/repository/MozillaSecurity/site-scout/main/badge.svg)](https://community-tc.services.mozilla.com/api/github/v1/repository/MozillaSecurity/site-scout/main/latest)
[![codecov](https://codecov.io/gh/MozillaSecurity/site-scout/branch/main/graph/badge.svg)](https://codecov.io/gh/MozillaSecurity/site-scout)
[![Matrix](https://img.shields.io/badge/dynamic/json?color=green&label=chat&query=%24.chunk[%3F(%40.canonical_alias%3D%3D%22%23fuzzing%3Amozilla.org%22)].num_joined_members&suffix=%20users&url=https%3A%2F%2Fmozilla.modular.im%2F_matrix%2Fclient%2Fr0%2FpublicRooms&style=flat&logo=matrix)](https://riot.im/app/#/room/#fuzzing:mozilla.org)

Site Scout is a tool that can identify and report issues that are triggered in the wild.
A URL or a collection of URLs to visit must be provided. The collection is iterated over
and each URL is visited. All results are collected and reported.

The primary goal is to locate issues such as crashes, assertion failures and other
issues detected by AddressSanitizer, UndefinedBehaviourSanitizer and ThreadSanitizer.

Usage
-----

Visit a collections of URLs:
```
python3 -m site_scout <firefox-bin> -i <urls.yml>
```

-or-

Visit one or more URLs (typically used to reproduce an issue):
```
python3 -m site_scout <firefox-bin> -u <url> [<url> ...]
```

Input Format
------------

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

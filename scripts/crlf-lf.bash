#!/usr/bin/env bash
# usage: ./scripts/crlf-lf.bash

# one file at a time using 8 threads
find . -type f -not -path "./target/*" -print0 | xargs -0 -n 1 -P 8 dos2unix

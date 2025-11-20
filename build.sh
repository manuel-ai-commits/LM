#!/bin/sh

git submodule update --remote --recursive

set -x

cc -Wall -W -Wextra -O2 lm.c
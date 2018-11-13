#!/bin/bash
set -e
exec pytest --mpl-oggm --mpl-upload --pyargs oggm

#!/usr/bin/env bash

poetry version patch && DISTWHL=`poetry build | awk -F " " -v x=3 '/[^ ]?\.whl/ {print $x}'`; pip install --no-cache-dir --force-reinstall "dist/$DISTWHL"

#!/bin/bash

DIFF=`yapf -p -r -d --style='{COLUMN_LIMIT:100}' -e third_libs/MegCityProto -e tools/evaluation ./`
if [ ! -z "$DIFF" ]
then
    echo "yapf format check failed"
    printf -- "$DIFF"
    false
else
    echo "yapf format check succeeded"
fi

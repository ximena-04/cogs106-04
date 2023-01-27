#!/bin/bash

date > version

git add update version.sh version

git commit -m "?"

git push

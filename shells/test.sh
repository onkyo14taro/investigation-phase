#!/bin/bash

dir_project="$(dirname $(cd $(dirname $0); pwd))"
cd "${dir_project}"


pytest
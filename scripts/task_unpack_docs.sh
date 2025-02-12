#!/bin/bash
# Extract a docs.tgz from TVM's docs build and stage for a git push
set -euxo pipefail

DOCS_TGZ=$1
DEPLOY_BRANCH=$2

cleanup()
{
    git checkout main
}
trap cleanup 0

if [ ! -f "$DOCS_TGZ" ]; then
    echo "$DOCS_TGZ does not exist"
    exit 255
fi

git checkout -B "$DEPLOY_BRANCH" "origin/$DEPLOY_BRANCH"
rm -rf docs
mkdir docs
tar xf "$DOCS_TGZ" -C docs

#!/bin/bash
# Deploy the website to the asf-site branch.
set -e
set -u

echo "Start to generate and deploy site ..."
bundle exec jekyll b
cp .gitignore .gitignore.bak
cp .asf.yaml .asf.yaml.bak

# copy new files into the current site
git fetch
git checkout -B asf-site origin/asf-site

# remove all existing files, excluding the docs
git ls-files | grep -v ^docs| xargs  rm -f
cp .gitignore.bak .gitignore
cp .asf.yaml.bak .asf.yaml

cp -rf _site/* .
DATE=`date`
git add --all && git commit -am "Build at ${DATE}"
git push origin asf-site
git checkout main
echo "Finish deployment at ${DATE}"

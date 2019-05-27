#! /bin/sh
mkdir bs_includes || (echo "Skipping SETUP of BS_INCLUDES" && exit 0)
cd  bs_includes
waf setup --project=pyhmf --repo-db-url=http://github.com/electronicvisions/projects

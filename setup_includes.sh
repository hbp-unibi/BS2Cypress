#! /bin/sh
mkdir bs_includes || (cd bs_includes && waf repos-update && exit 0)
cd  bs_includes
waf setup --project=pyhmf --repo-db-url=http://github.com/electronicvisions/projects

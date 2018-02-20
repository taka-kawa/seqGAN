#!/bin/sh
# ref: http://dev.syosetu.com/man/api/
LIST=$(seq -w 1 100 2000)
URL='http://api.syosetu.com/novelapi/api/?gzip=1&out=json&of=s&lim=100'
GENRE="101 102 201 202 301 302 303 304 305 306 307 401 402 403 404 9901 9902 9903 9904 9999 9801"
for genre in $GENRE
do
    for i in $LIST
    do
    curl $URL"&genre=$genre&st=$i" |gzip -cd > g$genre-$i.json
    sleep 40
    done
done

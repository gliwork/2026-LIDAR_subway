#!/bin/sh
#  ============================================================================
#   Keeps  docker/src/  in  sync  with  the  working  .py  scripts  that  live
#   in  the  parent  folder  .  Run  it  any  time  after  you  edit  a  script
#   and  before  you  rebuild  /  push  the  docker  folder  :
#
#       docker/sync-src.sh
#
#   It  copies  every  <parent>/*.py  into  docker/src/  (  overwriting  )  .
#  ============================================================================
set  -eu
HERE="$(cd  "$(dirname  "$0")"  &&  pwd)"
PARENT="$(cd  "$HERE/.."  &&  pwd)"
mkdir  -p  "$HERE/src"
cp  -f  "$PARENT"/*.py  "$HERE/src/"
echo  "synced  $(ls  -1  "$HERE/src"  |  wc  -l  )  .py  files  from  $PARENT  ->  $HERE/src"

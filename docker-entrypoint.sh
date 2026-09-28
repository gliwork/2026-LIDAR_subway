#!/bin/sh
#  camera_online.py  entrypoint.
#
#  If  the  host  X  display  is  exported  into  the  container
#  (  -e  DISPLAY  )  the  Qt  window  is  shown  on  the  host.
#  Otherwise  start  a  local  Xvfb  (  no  auth  needed  inside  the
#  container  )  :  the  full  pipeline  still  runs  and  prints  the
#  per-frame  distances  ,  the  window  itself  is  invisible  .
set -e

if [ -n "$DISPLAY" ]; then
    echo "display = $DISPLAY"
    exec python3 camera_online.py "$@"
else
    echo "no \$DISPLAY - starting a virtual Xvfb display (window not visible)"
    Xvfb :99 -screen 0 1280x720x24 >/dev/null 2>&1 &
    XVFB_PID=$!
    sleep 1
    DISPLAY=:99 python3 camera_online.py "$@"
    RC=$?
    kill "$XVFB_PID" 2>/dev/null
    exit "$RC"
fi

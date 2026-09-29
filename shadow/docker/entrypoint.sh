#!/bin/sh
#  ============================================================================
#   LCT  2026  entrypoint
#
#   The  input  lidar  ROS  bags  are  ALWAYS  read  from  /data  .  This  script
#   finds  the  bag  for  you  (  the  first  .db3  /  .zst  in  /data  )  unless
#   you  pass  an  explicit  file  path  as  the  first  argument  .  Outputs
#   go  to  /out  unless  you  pass  --out  .
#
#   usage  (  from  the  host  )  :
#
#     #  auto  -  detect  the  bag  in  /data  ,  default  flags  :
#     docker  run  --rm  -v  /host/bags:/data  -v  /host/results:/out  lct-shadow
#
#     #  extra  flags  pass  straight  through  to  the  tool  :
#     docker  run  --rm  -v  /host/bags:/data  -v  /host/results:/out  lct-shadow \
#         --d0  30  --window  5  --excl  --video-scale  8
#
#     #  point  at  a  specific  bag  (  inside  /data  ,  or  any  path  )  :
#     docker  run  --rm  -v  /host/bags:/data  -v  /host/results:/out  lct-shadow \
#         /data/doubleT_platform_0.db3  --d0  30
#
#     #  use  the  two  -  pass  consensus  tool  instead  of  the  streaming  one  :
#     TOOL=rosbag_shadow  docker  run  --rm  -v  /host/bags:/data  -v  /host/results:/out  lct-shadow
#  ============================================================================
set  -eu

#  ----  resolve  the  input  bag  --------------------------------------------
if  [  "$#"  -gt  0  ]  &&  [  -f  "$1"  ]  ;  then
    BAG="$1"
    shift
else
    BAG="$(ls  -1  /data/*.db3  /data/*.zst  2>/dev/null  |  head  -n1  ||  true)"
    if  [  -z  "$BAG"  ]  ;  then
        echo  "lct-entry  :  no  .db3  /  .zst  bag  found  in  /data"  >&2
        echo  "  mount  your  bags  with  :  -v  /host/bags:/data"  >&2
        echo  "  or  pass  a  bag  path  as  the  first  argument  ."  >&2
        exit  1
    fi
fi
echo  "lct-entry  :  processing  bag  ->  $BAG"

#  ----  default  output  dir  (  only  if  the  user  did  not  set  --out  )  --
if  !  printf  '%s\n'  "$@"  |  grep  -q  --  '^\-\-out[= ]'  ;  then
    set  --  "$@"  --out  /out
fi

#  ----  pick  the  tool  (  TOOL  env  var  ,  default  =  streaming  )  -------
TOOL="${TOOL:-rosbag_shadow_stream}"
case  "$TOOL"  in
    rosbag_shadow_stream)  SCRIPT="/app/rosbag_shadow_stream.py"  ;;
    rosbag_shadow)         SCRIPT="/app/rosbag_shadow.py"         ;;
    *)                     SCRIPT="/app/${TOOL}.py"               ;;
esac
if  [  !  -f  "$SCRIPT"  ]  ;  then
    echo  "lct-entry  :  TOOL  '$TOOL'  ->  no  such  script  $SCRIPT"  >&2
    exit  1
fi

exec  python  "$SCRIPT"  "$BAG"  "$@"

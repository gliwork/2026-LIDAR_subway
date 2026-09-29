#  ============================================================================
#   LCT  2026  -  lidar  /  rosbag  shadow  -  gram  processing
#
#   SELF  -  CONTAINED  :  this  folder  (  docker/  )  holds  everything  the
#   build  needs  (  Dockerfile  +  src/  +  requirements  +  entrypoint  )  ,
#   so  it  can  be  dropped  into  any  repo  as  a  sub  -  folder  and  built
#   with  the  folder  itself  as  the  build  context  .
#
#   The  input  lidar  ROS  bags  (  .db3  /  .zst  )  are  ALWAYS  read  from
#   the  fixed  container  path  /data  .  Outputs  go  to  /out  .
#
#   build  (  from  the  parent  of  this  folder  )  :
#       docker  build  -t  lct-shadow  docker/
#   ...  or  equivalently  :
#       cd  docker  &&  docker  build  -t  lct-shadow  .
#
#   run  (  mount  your  bags  into  /data  and  a  results  dir  into  /out  )  :
#       docker  run  --rm  -v  /host/bags:/data  -v  /host/results:/out  lct-shadow
#
#   extra  flags  pass  straight  through  to  the  tool  :
#       docker  run  --rm  -v  /host/bags:/data  -v  /host/results:/out  lct-shadow \
#           --d0  30  --window  5  --excl
#
#   pick  the  tool  with  the  TOOL  env  var  (  default  =  streaming  )  :
#       TOOL=rosbag_shadow  docker  run  --rm  -v  ...  lct-shadow
#  ============================================================================

FROM  python:3.11-slim

#  system  deps  :
#    ffmpeg   -  the  CLI  the  scripts  pipe  raw  frames  into  for  mp4  ;
#    zstd     -  needed  to  stream  -  extract  a  .zst  (  zstd  -  tar  )  bag  ;
#    ca  -  certificates  -  for  pip  over  https  .
RUN  apt-get  update  \
    &&  apt-get  install  -y  --no-install-recommends  \
           ffmpeg  zstd  ca-certificates  \
    &&  rm  -rf  /var/lib/apt/lists/*

#  python  deps  (  pinned  -  see  requirements.txt  )
COPY  requirements.txt  /tmp/requirements.txt
RUN  pip  install  --no-cache-dir  -r  /tmp/requirements.txt  \
    &&  rm  /tmp/requirements.txt

#  headless  matplotlib  backend  +  unbuffered  logs
ENV  MPLBACKEND=Agg  \
    PYTHONUNBUFFERED=1  \
    PIP_NO_CACHE_DIR=1

#  the  project  scripts  (  every  .py  in  src/  )
WORKDIR  /app
COPY  src/*.py  /app/

#  the  fixed  input  +  output  locations  the  whole  image  assumes
RUN  mkdir  -p  /data  /out

#  entrypoint  :  resolves  the  bag  in  /data  and  launches  the  tool
COPY  entrypoint.sh  /usr/local/bin/lct-entry
RUN  chmod  +x  /usr/local/bin/lct-entry

ENTRYPOINT  [  "/usr/local/bin/lct-entry"  ]
#  default  :  show  -  all  canvas  +  cumulative  top  -  down  map  ,  D0  =  30  m
CMD  [  "--d0",  "30",  "--canvas",  "full",  "--topdown"  ]

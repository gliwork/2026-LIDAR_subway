#  ============================================================
#  laser-camera  container  :  camera_online.py
#
#  Build  (  from  the  directory  that  contains  this  file  )  :
#      docker  build  -t  laser-cam  .
#
#  Run  with  the  host  USB  camera  :
#      docker  run  --rm  -it  --device  /dev/video0  laser-cam
#      docker  run  --rm  -it  --device  /dev/video1  laser-cam  --cam  1
#      docker  run  --rm  -it  --device  /dev/video0  laser-cam  --quiet
#
#  To  SEE  the  window  on  the  host  (  X11  host  only  )  :
#      docker  run  --rm  -it  --device  /dev/video0  \
#          -e  DISPLAY=$DISPLAY  \
#          -v  $HOME/.Xauthority:/root/.Xauthority:ro  \
#          laser-cam
#
#  Replay  a  video  instead  of  the  camera  (  no  device  needed  )  :
#      docker  run  --rm  -it  -v  $PWD/videos:/data  laser-cam  \
#          --video  /data/  some.mp4
#
#  Headless  host  (  no  X  )  :  the  entrypoint  automatically
#  falls  back  to  a  virtual  framebuffer  (  Xvfb  )  ;  the
#  measurement  still  runs  and  prints  per-frame  distances,
#  but  the  window  is  not  visible.
#
#  Non-root  camera  access  :  add  --group-add  video  (  or
#  --user  "$(id  -u  )  :$(id  -g  )"  --group-add  video  )  .
#  ============================================================
FROM  python:3.11-slim-bookworm

ENV  DEBIAN_FRONTEND=noninteractive

#  System  libraries  the  opencv-python  wheel  needs  :
#    libglib2.0-0  libsm6  libxext6  libxrender1  libgomp1  libgl1
#        ->  highgui  (  Qt  window  )  +  shared  libs
#    xvfb  ->  virtual  display  fallback  on  headless  machines
#  (  the  V4L2  USB-camera  backend  is  built  into  the  wheel  )
RUN  apt-get  update  &&  apt-get  install  -y  --no-install-recommends  \
    xvfb  \
        libglib2.0-0  \
        libgl1  \
        libgomp1  \
        libsm6  \
        libxext6  \
        libxrender1  \
    &&  rm  -rf  /var/lib/apt/lists/*

WORKDIR  /laser

#  dependencies  first  (  the  layer  that  changes  least  often  )  ;
#  requirements.txt  is  for  bare  (  non-docker  )  installs  and  is
#  dropped  from  the  image  after  the  install
COPY  requirements.txt  /tmp/requirements.txt
RUN  pip  install  --no-cache-dir  -r  /tmp/requirements.txt  &&  rm  /tmp/requirements.txt

#  the  app  and  its  local  imports  (  the  absolute  minimum  :  the
#  entry  script  +  the  two  modules  it  imports  +  the  calibration  )
COPY  camera_online.py  measure.py  laser_sensor.py  ./

#  calibration.json  :  keep  only  what  load_calibration  actually
#  reads  (  "model"  +  "maps"  )  ;  drop  the  calibrate.py  diagnostics
#  (  "report"  /  "meta"  )  .  ~9.6  KB  ->  ~4.9  KB  .
COPY  calibration.json  ./
RUN  python3  -c  "import  json;  d  =  json.load(open('calibration.json'));  d.pop('report',  None);  d.pop('meta',  None);  json.dump(d,  open('calibration.json','w'),  indent=1)"

COPY  docker-entrypoint.sh  .
RUN  chmod  +x  docker-entrypoint.sh

ENTRYPOINT  [  "./docker-entrypoint.sh"  ]
CMD  [  "--cam",  "0"  ]

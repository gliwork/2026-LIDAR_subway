#  laser  —  three-laser  USB-camera  distance  measurement  (  live  )

`camera_online.py`  reads  frames  from  a  USB  camera  and  measures
the  distance  to  the  closest  obstacle  in  front  of  the  vehicle
from  the  reflections  of  three  laser  lines  (  L  /  R  /  B  fans
)  .  The  result  is  shown  in  a  live  window  :

  *  big  color-coded  stamp  :  MIN  x.xx  m  (  green  ≥  3  m  ,
     yellow  1.5–3  m  ,  red  <  1.5  m  ,  gray  =  no  laser  seen  )
  *  per-laser  readings  L  /  R  /  B
  *  vanishing-point  corner  trajectories
  *  yellow  boxes  around  near  obstacles  (  confirmed  over  3
     consecutive  frames  )

The  measurement  runs  on  the  1280×720  reference  frame  ;  other
camera  resolutions  are  resized  automatically  .

##  Files  (  the  minimum  set  )

|  file  |  role  |
|---|---|
|  `camera_online.py`  |  the  live  application  (  entry  point  )  |
|  `measure.py`  |  per-frame  measurement  +  annotation  (  imported  )  |
|  `laser_sensor.py`  |  laser-line  model  ,  calibration  I/O  (  imported  )  |
|  `calibration.json`  |  calibrated  model  +  line  maps  (  default  `--cal`  )  |
|  `requirements.txt`  |  Python  dependencies  (  bare  install  )  |
|  `Dockerfile`  +  `docker-entrypoint.sh`  |  portable  container  (  see  below  )  |

Only  `numpy`  and  `opencv-python`  (  GUI  build  )  are  needed
—  nothing  else  .

##  Run  on  any  computer  (  no  Docker  )

```bash
python3  -m  venv  venv
.  venv/bin/activate        #  Windows:  venv\Scripts\activate
pip  install  -r  requirements.txt
python3  camera_online.py
```

###  Options

```
--cam  N               camera  index  (  default  0  )
--video  FILE.mp4      replay  a  video  instead  of  the  camera
--cal  FILE.json       calibration  (  default  calibration.json  )
--width  /  --height   requested  camera  resolution  (  1280x720  )
--min-brightness  F    raise  it  in  bright  scenes
--quiet                no  per-frame  console  log
```

Keys  :  `q`  or  `ESC`  to  quit  .

###  Notes

  *  Many  UVC  cameras  deliver  a  given  resolution  only  with  the
     MJPG  fourcc  —  the  script  requests  it  automatically  ;  the
     real  frame  rate  is  what  the  camera  can  give  (  ~10–15  fps
     at  1280×720  for  typical  sensors  )  .
  *  Linux  :  the  camera  is  accessed  via  V4L2  (`/dev/videoN`  )
     ;  the  user  must  belong  to  the  `video`  group  (  or  run  as
     root  )  .
  *  `--calibration`  was  obtained  from  28  dark  scene  images
     covering  1.955–13.2  m  (  see  `calibrate.py`  /  the
     calibration  folder  )  ;  27/28  images  reproduce  within
     0.072  m  .  If  the  camera  /  laser  geometry  changes  ,
     re-run  `calibrate.py`  and  pass  the  new  file  with  `--cal`  .

##  Docker  (  portable  ,  ~640  MB  image  )

The  image  contains  only  the  files  above  (  112  KB  in  `/laser`
)  :  the  entry  script  ,  the  two  imported  modules  and  a  slimmed
`calibration.json`  (  `model`  +  `maps`  —  exactly  what
`load_calibration()`  reads  ;  the  `calibrate.py`  diagnostics  are
stripped  at  build  time  )  .  System  side  :  Python  3.11  slim  +
the  OpenCV  Qt  libraries  +  `Xvfb`  (  virtual  display  fallback  )  .

```bash
docker  build  -t  laser-cam  .
```

###  Run  variants

```bash
#  headless  :  pipeline  runs  under  Xvfb  ,  distances  on  the
#  console  ,  no  window
docker  run  --rm  -it  --device  /dev/video0  laser-cam

#  window  on  the  host  (  X11  host  only  )
docker  run  --rm  -it  --device  /dev/video0  \
    -e  DISPLAY=$DISPLAY  \
    -v  $HOME/.Xauthority:/root/.Xauthority:ro  \
    laser-cam

#  another  camera  /  options  pass  through  as-is
docker  run  --rm  -it  --device  /dev/video1  laser-cam  --cam  1  --quiet

#  replay  a  video  (  no  camera  needed  )
docker  run  --rm  -it  -v  $PWD/videos:/data  laser-cam  \
    --video  /data/  some.mp4
```

Non-root  container  users  need  `--group-add  video`  (  the  default
root  user  works  out  of  the  box  )  .

###  Verified

  *  inside  the  container  the  slimmed  calibration  loads  and  a
     real  calibration  image  measures  L  =  1.955  ,  R  =  1.957  ,
     B  =  1.955  m  (  the  1.955  m  ground-truth  frame  of  the
     calibration  series  )  ;
  *  the  Qt  window  initializes  under  the  built-in  Xvfb  without
     auth  ;
  *  with  no  camera  device  attached  the  run  exits  cleanly  with
     `cannot  open  camera  0  (  check  the  connection  and  the
     index  )`  .

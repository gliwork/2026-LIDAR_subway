#  LCT  2026  -  Docker  image  for  lidar  /  rosbag  shadow  -  gram  processing

A  small  ,  **self  -  contained **  folder  that  builds  a  Docker  image
running  the  shadow  -  gram  tools  on  your  lidar  ROS  bags  .  Everything
the  build  needs  lives  *inside  this  folder  *  ,  so  you  can  drop  it
into  any  GitHub  repo  as  a  sub  -  folder  and  build  it  with  one
command  -  no  files  are  needed  outside  it  .

**Convention  :**  the  input  bags  are  **always**  read  from  the  fixed
container  path  **`/data`**  ;  everything  the  tools  write  goes  to
**`/out`**  .  You  never  hard  -  code  a  host  path  -  you  just  mount  a
folder  into  `/data`  .

---

##  1  .  Layout

```
docker/
   Dockerfile          #  the  image  (  build  context  =  this  folder  )
   .dockerignore       #  keeps  the  build  context  to  sources  only
   requirements.txt    #  pinned  python  deps
   entrypoint.sh       #  finds  the  bag  in  /data  ,  sets  --out  /out  ,  runs  the  tool
   sync-src.sh         #  re  -  copies  the  parent  .py  files  into  src/
   src/                #  the  .py  scripts  the  image  runs  (  a  copy  of
      ...16  .py       #  the  working  scripts  from  the  parent  folder  )
   README.md
```

Because  `src/`  is  a  **copy**  of  the  working  scripts  ,  run
`./sync  -  src.sh`  (  see  §  3  )  after  editing  a  script  and  before
you  rebuild  or  push  this  folder  .

---

##  2  .  Build

From  the  **parent  of  this  folder  **  (  the  build  context  is
`docker/`  itself  )  :

```bash
docker  build  -t  lct-shadow  docker/
```

...  or  equivalently  :

```bash
cd  docker  &&  docker  build  -t  lct-shadow  .
```

The  build  installs  `ffmpeg`  +  `zstd`  and  the  six  pinned  python
packages  (  numpy  /  scipy  /  matplotlib  /  opencv  -  headless  /  Pillow  /
laspy  )  ,  then  copies  `src/*.py`  into  the  image  at  `/app`  .

---

##  3  .  Keeping  `src/`  in  sync

The  scripts  you  actually  edit  live  in  the  **parent**  folder  .  After
changing  one  ,  refresh  the  copy  the  image  uses  :

```bash
docker/sync  -  src.sh          #  copies  <parent>/*.py  ->  docker/src/
docker  build  -t  lct-shadow  docker/
```

(  On  a  machine  where  this  folder  is  the  canonical  copy  -  e.g.  inside
the  GitHub  repo  -  just  edit  the  files  in  `src/`  directly  ;  the
sync  script  is  only  for  the  original  working  tree  .  )

---

##  4  .  Run

Mount  a  host  folder  holding  your  `.db3`  /  `.zst`  bags  onto  `/data`
and  a  results  folder  onto  `/out`  :

```bash
docker  run  --rm  \
    -v  /path/to/bags:/data  \
    -v  /path/to/results:/out  \
    lct-shadow
```

*  **`/data`**  -  input  .  The  entrypoint  picks  the  first  `.db3`  /
    `.zst`  it  finds  there  .  `.zst`  bags  (  zstd  -  tar  )  are  auto
    -  extracted  to  a  temp  `.db3`  and  cleaned  up  afterwards  .
*  **`/out`**  -  output  :  `shadow_stream.mp4`  (  or  the  non  -  streaming
    outputs  )  ,  `frames/`  ,  `masks/`  ,  `topdown_map.png`  ,
    `topdown_video.mp4`  ,  `summary.json`  .

Several  bags  in  `/data`  ?  Point  at  one  explicitly  :

```bash
docker  run  --rm  -v  /path/to/bags:/data  -v  /path/to/results:/out  lct-shadow \
    /data/doubleT_platform_0.db3
```

---

##  5  .  Flags  (  pass  -  through  to  the  tool  )

Everything  after  the  image  name  is  forwarded  to  the  python  tool  ,  so
all  of  its  flags  work  unchanged  :

```bash
docker  run  --rm  -v  /path/to/bags:/data  -v  /path/to/results:/out  lct-shadow \
    --d0  30  --res  0.10  --window  5  --buffer  10  --excl  \
    --canvas  full  --canvas-sample  1  --topdown  --td-video  topdown_video.mp4 \
    --fps  10  --video-scale  8  --min-px  6
```

The  default  `CMD`  is  `--d0  30  --canvas  full  --topdown`  ,  i.e.  the
**show  -  all  canvas  +  cumulative  top  -  down  tunnel  map**  (  the
curved  /  fork  -  aware  mode  )  .

###  Choosing  the  tool

The  entrypoint  runs  `rosbag_shadow_stream.py`  by  default  (  single
forward  pass  -  best  for  big  bags  )  .  Switch  with  the  `TOOL`  env  var
(  any  script  in  `src/`  works  )  :

```bash
TOOL=rosbag_shadow   docker  run  --rm  -v  ...  lct-shadow   #  two  -  pass  consensus
TOOL=lidar_view      docker  run  --rm  -v  ...  lct-shadow   #  3  -  D  point  -  cloud  view
TOOL=shadow_detect   docker  run  --rm  -v  ...  lct-shadow
```

---

##  6  .  What  is  inside

*  base  `python:3.11-slim`  (  Debian  )  ;
*  `ffmpeg`  (  encodes  the  mp4  from  a  raw  frame  pipe  )  and  `zstd`
    (  opens  `.zst`  bags  )  ;
*  python  :  `numpy  2.2.0`  ,  `scipy  1.14.1`  ,  `matplotlib  3.10.8`  ,
    `opencv  -  python  -  headless  4.13.0`  ,  `Pillow  11.0.0`  ,
    `laspy  2.7.0`  (  same  versions  as  the  host  `llm`  env  )  ;
*  every  `src/*.py`  in  `/app`  ;
*  `MPLBACKEND=Agg`  (  headless  matplotlib  )  .

Both  bag  schemas  are  supported  :  the  MCROS  v3  `topics`  +
`messages  (  topic_id  ,  ...  )`  layout  and  the  older  layout  with  the
topic  name  stored  directly  in  `messages`  .  An  empty  /  corrupt  bag
fails  with  a  clear  message  .

---

##  7  .  Notes  /  caveats

*  **Output  file  ownership  .**  The  container  runs  as  `root`  by
    default  ,  so  everything  it  writes  into  `/out`  is  owned  by  `root`
    (  world  -  readable  ,  so  you  can  always  view  or  copy  the  videos
    and  maps  ;  but  a  non  -  root  host  user  can  not  delete  or
    overwrite  them  without  `sudo`  )  .  To  own  the  outputs  ,  run  with
    your  own  uid  (  the  mounted  folders  must  be  accessible  to  it  )  :

    ```bash
    docker  run  --rm  --user  "$(id  -u):$(id  -g)"  \\
        -v  /path/to/bags:/data  -v  /path/to/results:/out  lct-shadow
    ```

*  **Symlinks  .**  A  bag  may  be  a  symlink  ,  but  its  **target  must
    be  inside  a  mounted  folder  **  -  the  `zstd`  CLI  will  not  follow
    a  link  pointing  outside  the  container  .  The  script  resolves  links
    and  tells  you  clearly  when  the  target  is  not  reachable  .

*  **`.zst`  bags  .**  A  zstd  -  tar  bag  is  extracted  to  a  temporary
    `.db3`  inside  the  container  (  a  few  GB  of  free  space  in  the
    docker  data  root  )  and  removed  again  when  the  run  finishes  .

*  **Shipping  the  image  binary  .**  Anyone  with  Docker  can  build  it
    from  this  folder  .  If  you  must  hand  over  the  built  image  to
    someone  without  Docker  ,  use  `docker  save  lct-shadow  -o
    lct-shadow.tar`  and  share  the  `.tar`  (  e.g.  as  a  GitHub  Release
    asset  )  -  do  **not**  commit  the  `.tar`  into  the  repo  .

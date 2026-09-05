# Backend / frontend contract (v1)
Backend owner may refine this by messaging primary before incompatible changes. JSON endpoints:
- GET /api/health -> status/runtime summary.
- GET /api/experiments -> real run summaries and learning curves; no synthetic metrics.
- GET /api/replays -> list {id,label,kind,seed,success,steps,stage}; GET /api/replays/{id} -> {id,label,kind,seed,success,steps,frames:[state],actions:[],metrics:{}}.
- POST /api/play -> {seed,stage,checkpoint?,grid?} starts/executes a policy episode and returns replay shape. CPU model inference; error clearly if no model. Support scripted baseline explicitly labeled.
- POST /api/validate -> {grid: array of 9 ASCII strings} -> {valid,errors:[]}. Allowed cells # wall, . floor, S start, E exit, K key, D locked door, B banana, C camera. Validate counts/bounds/solvability.
State shape: {grid:[ASCII strings],agent:{x,y,dir},inventory:{key:bool,banana:bool},camera?:{x,y,dir},visible:[[x,y],...],step,max_steps,stage,status,reward?}. dir 0 east,1 south,2 west,3 north. Grid static features plus updated removed keys/banana/open door; lowercase d open door. Camera state separate. Frames include initial and every subsequent state. Status running/escaped/caught/timeout. All top-level state renderer data can be privileged, policy observations MUST remain partial.
Static frontend at / from repo web. Data artifacts under artifacts/data; models in runs (ignored) with release plan primary-owned. API never starts training. Validate input and bound inference episodes.

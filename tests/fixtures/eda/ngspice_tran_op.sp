RC low pass
V1 in 0 PULSE(0 1 0 1p 1p 5m 10m)
R1 in out 1k
C1 out 0 1u
* (run from .control by Grafux) .tran 200u 4m
* (run from .control by Grafux) .op
* --- measurements (meas_statements port) ---
.meas tran trise TRIG v(out) VAL=0.1 RISE=1 TARG v(out) VAL=0.9 RISE=1
.meas tran never WHEN v(out)=5
* --- simulation control (added by Grafux) ---
.control
set filetype=ascii
set appendwrite
tran 200u 4m
write sim.raw
op
write sim.raw
quit
.endc
.end

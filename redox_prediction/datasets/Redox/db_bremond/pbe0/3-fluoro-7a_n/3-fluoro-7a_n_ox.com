%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 3-fluoro-7a_n_ox

 1 2
 C       -1.0392770000       -3.9980300000       -0.0000000000
 C       -2.3223940000       -3.4216120000       -0.0000000000
 C        0.0858160000       -3.1989740000       -0.0000000000
 C       -0.0417420000       -1.7958160000       -0.0000000000
 C       -1.3357310000       -1.2175260000       -0.0000000000
 C       -2.4686770000       -2.0454720000       -0.0000000000
 H       -0.9294010000       -5.0843220000       -0.0000000000
 H        1.0834890000       -3.6458530000       -0.0000000000
 H       -3.2084380000       -4.0594080000       -0.0000000000
 H       -3.4645470000       -1.5956000000       -0.0000000000
 N        1.0852620000       -1.0186100000       -0.0000000000
 C        1.1647460000        0.3467430000        0.0000000000
 C       -0.0000000000        1.1578090000        0.0000000000
 S       -1.6039420000        0.4972040000        0.0000000000
 C        2.4324440000        0.9662020000        0.0000000000
 C        2.5442420000        2.3386560000        0.0000000000
 C        1.3779530000        3.1224640000        0.0000000000
 C        0.1171350000        2.5526240000        0.0000000000
 H        3.3325150000        0.3460200000       -0.0000000000
 H        3.5180910000        2.8317720000        0.0000000000
 H       -0.7639490000        3.1979470000        0.0000000000
 H        1.9707860000       -1.5181550000       -0.0000000000
 F        1.4934000000        4.4289120000        0.0000000000


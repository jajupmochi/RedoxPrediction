%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 3l_2c_gn

 2 1
 N        0.0000000000        0.0000000000       -3.4836350000
 C       -0.4322570000        1.0993660000       -2.8444550000
 C       -0.4378780000        1.1254570000       -1.4604050000
 C        0.4322570000       -1.0993660000       -2.8444550000
 C        0.4378780000       -1.1254570000       -1.4604050000
 C        0.0000000000        0.0000000000       -0.7438080000
 C       -0.0000000000        0.0000000000        0.7438080000
 C        0.4378780000        1.1254570000        1.4604050000
 C       -0.4378780000       -1.1254570000        1.4604050000
 C       -0.4322570000       -1.0993660000        2.8444550000
 C        0.4322570000        1.0993660000        2.8444550000
 N       -0.0000000000        0.0000000000        3.4836350000
 H       -0.7688900000       -1.9369750000        3.4602950000
 H       -0.8107150000       -2.0202080000        0.9576480000
 H        0.8107150000        2.0202080000        0.9576480000
 H        0.7688900000        1.9369750000        3.4602950000
 H        0.8107150000       -2.0202080000       -0.9576480000
 H        0.7688900000       -1.9369750000       -3.4602950000
 H       -0.7688900000        1.9369750000       -3.4602950000
 H       -0.8107150000        2.0202080000       -0.9576480000
 H        0.0000000000        0.0000000000       -4.5073820000
 H       -0.0000000000        0.0000000000        4.5073820000

%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 14-nq_n_rd

-1 2
 C        2.6554450000       -0.7043570000        0.0000070000
 C        1.4545320000       -1.3933530000       -0.0000010000
 C        0.2243750000       -0.7097860000       -0.0000090000
 C        0.2243750000        0.7097860000       -0.0000080000
 C        1.4545320000        1.3933530000        0.0000000000
 C        2.6554450000        0.7043570000        0.0000070000
 H        3.6039100000        1.2511860000        0.0000130000
 H        3.6039100000       -1.2511860000        0.0000120000
 H        1.4065670000       -2.4856880000       -0.0000040000
 H        1.4065670000        2.4856880000       -0.0000000000
 C       -1.0317000000       -1.4849370000       -0.0000210000
 C       -2.2311990000       -0.6901800000        0.0000030000
 C       -1.0317000000        1.4849370000       -0.0000170000
 C       -2.2311990000        0.6901800000        0.0000040000
 O       -1.0334030000       -2.7359800000        0.0000110000
 O       -1.0334030000        2.7359800000        0.0000070000
 H       -3.1719700000        1.2508400000        0.0000200000
 H       -3.1719700000       -1.2508390000        0.0000200000


%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT(calcall) FREQ SCF(tight) INT(ultrafine)

 p-cyanophenol_n_rd

-1 2
 C        0.0126800000       -0.0094700000       -0.2366200000
 C        0.0015500000        1.2039600000        0.4389900000
 C       -0.0046900000        1.2239700000        1.8393000000
 O       -0.0153800000        2.3871000000        2.5601800000
 C        0.0000400000        0.0246600000        2.5621500000
 C        0.0113200000       -1.1856200000        1.8859900000
 H        0.0151700000       -2.1176100000        2.4417900000
 C        0.0176200000       -1.2187300000        0.4786800000
 C        0.0293600000       -2.4676500000       -0.2153400000
 H        0.0177200000       -0.0275800000       -1.3215900000
 H       -0.0176000000        3.1413300000        1.9470900000
 H       -0.0050500000        0.0606200000        3.6467000000
 H       -0.0021600000        2.1373900000       -0.1201000000
 N        0.0336300000       -3.4860400000       -0.7782600000


%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT(calcall) FREQ SCF(tight) INT(ultrafine)

 3355-tchlorodiphenq_n_rd

-1 2
 C       -0.0000000000       -2.8461230000        1.2074610000
 C       -0.0000000000       -3.6700240000       -0.0000000000
 C       -0.0000000000       -1.4761030000        1.2072560000
 C       -0.0000000000       -0.7213530000        0.0000010000
 C       -0.0000000000       -1.4761020000       -1.2072540000
 C       -0.0000000000       -2.8461220000       -1.2074610000
 H       -0.0000000000       -0.9731320000        2.1741990000
 H       -0.0000000000       -0.9731300000       -2.1741980000
 C       -0.0000000000        0.7213530000        0.0000010000
 C       -0.0000000000        1.4761030000        1.2072560000
 C       -0.0000000000        2.8461230000        1.2074610000
 C       -0.0000000000        3.6700240000       -0.0000000000
 C       -0.0000000000        2.8461220000       -1.2074610000
 C       -0.0000000000        1.4761020000       -1.2072540000
 H       -0.0000000000        0.9731300000       -2.1741980000
 H       -0.0000000000        0.9731320000        2.1741990000
 O       -0.0000000000       -4.9003620000       -0.0000010000
 O       -0.0000000000        4.9003620000       -0.0000010000
Cl       -0.0000000000       -3.7076380000        2.7134140000
Cl       -0.0000000000       -3.7076360000       -2.7134150000
Cl       -0.0000000000        3.7076380000        2.7134140000
Cl       -0.0000000000        3.7076360000       -2.7134150000


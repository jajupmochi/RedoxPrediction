%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 12-nq_n_rd

-1 2
 C        3.0409670000        0.3165570000        0.0000020000
 C        2.0503870000        1.2830060000        0.0000010000
 C        0.6797920000        0.9359420000        0.0000000000
 C        2.6887380000       -1.0497900000       -0.0000000000
 C        1.3533790000       -1.4128010000       -0.0000040000
 C        0.3262160000       -0.4488470000       -0.0000040000
 H        3.4701790000       -1.8162190000       -0.0000010000
 H        1.0352130000       -2.4590850000       -0.0000070000
 H        4.0948190000        0.6124050000        0.0000040000
 H        2.3205610000        2.3448710000        0.0000020000
 C       -1.0748740000       -0.9001730000       -0.0000120000
 C       -2.1353650000        0.1587420000        0.0000300000
 C       -1.6817920000        1.5366220000        0.0000050000
 C       -0.3625440000        1.9134530000       -0.0000030000
 O       -1.3641170000       -2.1061140000        0.0000090000
 O       -3.3440350000       -0.1363310000       -0.0000190000
 H       -2.4795320000        2.2876040000       -0.0000010000
 H       -0.0854530000        2.9737110000       -0.0000110000


%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT(calcall) FREQ SCF(tight) INT(ultrafine)

 aniline_n_rd

-1 2
 C       -0.0799800000       -0.0588200000       -0.2517500000
 N       -0.0575400000       -0.0864200000       -1.6549300000
 C       -0.0832700000        1.1634600000        0.4402100000
 H       -0.1230900000        2.0959000000       -0.1193100000
 C       -0.0321800000        1.1869400000        1.8316600000
 H       -0.0381100000        2.1445100000        2.3461100000
 C        0.0241700000        0.0000300000        2.5646500000
 H        0.0630600000        0.0226100000        3.6494500000
 C        0.0273100000       -1.2165700000        1.8796600000
 H        0.0682000000       -2.1518900000        2.4320700000
 C       -0.0235500000       -1.2512100000        0.4884800000
 H       -0.0168400000       -2.2059800000       -0.0335200000
 H       -0.4280100000       -0.9397000000       -2.0592200000
 H       -0.4695100000        0.7304500000       -2.0928700000


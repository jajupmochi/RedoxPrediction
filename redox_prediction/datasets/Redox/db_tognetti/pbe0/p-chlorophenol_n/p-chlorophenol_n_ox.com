%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 p-chlorophenol_n_ox

 1 2
 C       -0.0000100000        0.0214400000       -0.1798500000
 C        0.0000000000        1.2326800000        0.5116900000
 C        0.0000100000        1.2428900000        1.9095600000
 O        0.0000000000        2.4054500000        2.6449000000
 C        0.0000100000        0.0370600000        2.6169300000
 C        0.0000000000       -1.1724400000        1.9282800000
 H        0.0000000000       -2.1113500000        2.4713600000
 C       -0.0000100000       -1.1733100000        0.5338800000
 H       -0.0000200000        0.0092800000       -1.2644000000
 H       -0.0000100000        3.1610200000        2.0345000000
 H        0.0000100000        0.0574700000        3.7021500000
 H        0.0000000000        2.1689900000       -0.0431500000
Cl       -0.0000200000       -2.7037600000       -0.3337100000


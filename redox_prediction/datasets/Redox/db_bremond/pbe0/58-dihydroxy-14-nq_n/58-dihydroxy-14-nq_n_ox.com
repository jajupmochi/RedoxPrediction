%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 58-dihydroxy-14-nq_n_ox

 1 2
 C       -1.0193310000       -2.3568730000       -0.0000390000
 C       -1.5781120000       -1.0408290000        0.0000600000
 C       -0.7139800000        0.1093930000        0.0000140000
 C        0.6547850000       -0.0911730000        0.0000070000
 C        1.2076330000       -1.4210270000       -0.0000270000
 C        0.3294310000       -2.5515500000       -0.0000500000
 H        0.7731290000       -3.5496600000       -0.0000700000
 H       -1.6986520000       -3.2147970000       -0.0000600000
 C       -1.2524440000        1.5169600000       -0.0001090000
 C       -0.2561320000        2.6161530000        0.0001160000
 C        1.0717400000        2.4078490000        0.0001780000
 C        1.6082320000        1.0502980000        0.0000520000
 O       -2.4352970000        1.7532130000       -0.0003890000
 O        2.8165330000        0.8240220000       -0.0000260000
 O        2.4767500000       -1.6151530000       -0.0000810000
 H        2.9232890000       -0.6857010000       -0.0000570000
 O       -2.8652180000       -0.8545870000        0.0002880000
 H       -3.3646870000       -1.6871460000        0.0002120000
 H        1.8013930000        3.2213520000        0.0002870000
 H       -0.6875480000        3.6207860000        0.0001610000


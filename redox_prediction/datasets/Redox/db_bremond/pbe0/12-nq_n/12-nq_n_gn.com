%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 12-nq_n_gn

 0 1
 C        3.0278690000        0.3539230000       -0.0000010000
 C        2.0075260000        1.3037950000        0.0000110000
 C        0.6657870000        0.9058440000        0.0000280000
 C        2.7228430000       -1.0076220000       -0.0000000000
 C        1.3913510000       -1.4178410000        0.0000150000
 C        0.3650310000       -0.4735510000        0.0000340000
 H        3.5250430000       -1.7490670000       -0.0000100000
 H        1.1162720000       -2.4751560000        0.0000210000
 H        4.0707170000        0.6800340000       -0.0000160000
 H        2.2513910000        2.3693680000        0.0000020000
 C       -1.0496000000       -0.9292540000        0.0000640000
 C       -2.1533610000        0.1584170000        0.0000730000
 C       -1.7186110000        1.5611080000        0.0000030000
 C       -0.4107900000        1.8912430000        0.0000110000
 O       -1.3636160000       -2.0944190000       -0.0000790000
 O       -3.3147240000       -0.1760660000       -0.0000900000
 H       -2.5077500000        2.3166360000       -0.0000460000
 H       -0.1172300000        2.9456890000       -0.0000220000


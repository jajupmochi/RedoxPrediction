%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 10-acetyl-7b_n_ox

 1 2
 C       -3.6491640000        0.3553730000       -0.2241450000
 C       -3.7630370000       -1.0384950000       -0.0876360000
 C       -2.4110430000        0.9665550000       -0.2288780000
 C       -1.2302940000        0.2029470000       -0.0904750000
 C       -1.3553180000       -1.2063430000        0.0328160000
 C       -2.6238740000       -1.8102700000        0.0347260000
 H       -4.5464010000        0.9666560000       -0.3407220000
 H       -2.3517230000        2.0458450000       -0.3745030000
 H       -4.7459330000       -1.5133600000       -0.0879020000
 H       -2.6977860000       -2.8964940000        0.1299430000
 N       -0.0000010000        0.8282090000       -0.0695020000
 C        1.2302990000        0.2029590000       -0.0904700000
 C        1.3553300000       -1.2063310000        0.0328180000
 S        0.0000100000       -2.2645870000        0.2068580000
 C        2.4110460000        0.9665740000       -0.2288510000
 C        3.6491700000        0.3553970000       -0.2241110000
 C        3.7630490000       -1.0384720000       -0.0876150000
 C        2.6238890000       -1.8102530000        0.0347340000
 H        2.3517230000        2.0458640000       -0.3744650000
 H        4.5464040000        0.9666860000       -0.3406720000
 H        2.6978050000       -2.8964770000        0.1299470000
 H        4.7459470000       -1.5133320000       -0.0878760000
 C       -0.0000130000        2.3299230000       -0.0075590000
 C       -0.0001180000        2.8587300000        1.3843930000
 H       -0.0002660000        3.9550170000        1.3598610000
 H        0.8846920000        2.4924160000        1.9287860000
 H       -0.8848020000        2.4921490000        1.9288060000
 O        0.0000820000        2.9401510000       -1.0241130000

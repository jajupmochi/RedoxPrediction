%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 44-dimethyl-22-bipyr_n_rd

-1 2
 C        2.8685980000        0.7565270000       -0.0000180000
 C        3.5171730000       -0.5040340000       -0.0000040000
 C        1.4891280000        0.7761130000       -0.0000180000
 C        2.6938450000       -1.6393340000        0.0000170000
 N        1.3786790000       -1.6393400000        0.0000220000
 C        0.7200280000       -0.4356910000        0.0000020000
 C       -0.7200290000       -0.4356920000        0.0000030000
 C       -1.4891280000        0.7761130000       -0.0000170000
 C       -2.8685980000        0.7565270000       -0.0000150000
 C       -3.5171730000       -0.5040340000        0.0000080000
 N       -1.3786790000       -1.6393400000        0.0000250000
 C       -2.6938450000       -1.6393340000        0.0000270000
 H        0.9849190000        1.7458920000       -0.0000340000
 H        4.6074100000       -0.5915530000       -0.0000120000
 H        3.1714240000       -2.6341740000        0.0000290000
 H       -0.9849180000        1.7458910000       -0.0000350000
 H       -3.1714240000       -2.6341740000        0.0000450000
 H       -4.6074100000       -0.5915530000        0.0000110000
 C        3.6725530000        2.0250430000        0.0000060000
 H        4.3320500000        2.0834750000       -0.8844680000
 H        4.3315850000        2.0837360000        0.8848120000
 H        3.0262010000        2.9162710000       -0.0002890000
 C       -3.6725530000        2.0250420000       -0.0000360000
 H       -3.0262010000        2.9162710000       -0.0000520000
 H       -4.3318160000        2.0836210000        0.8846030000
 H       -4.3318170000        2.0835910000       -0.8846770000

%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 14-nq-2-sulfonate_a_rd

-2 2
 C        3.6886210000       -1.5065420000        0.0001010000
 C        2.3239040000       -1.7441550000        0.0000310000
 C        1.3914700000       -0.6885280000       -0.0000530000
 C        1.8765970000        0.6483100000       -0.0000640000
 C        4.1679970000       -0.1776130000        0.0000890000
 C        3.2678990000        0.8738620000        0.0000070000
 C       -0.0539810000       -1.0156260000       -0.0001320000
 C       -0.9333570000        0.1437980000       -0.0000790000
 C       -0.4429340000        1.4323920000       -0.0000880000
 C        0.9524320000        1.7886810000       -0.0001580000
 H        4.3952760000       -2.3447750000        0.0001640000
 H        5.2471590000        0.0168250000        0.0001420000
 H        1.9040450000       -2.7545980000        0.0000330000
 H        3.5859010000        1.9210300000       -0.0000100000
 H       -1.1752600000        2.2445970000       -0.0000430000
 O       -0.4228610000       -2.2033170000        0.0000240000
 O        1.3706650000        2.9743260000       -0.0000320000
 S       -2.7424680000       -0.0756310000        0.0000300000
 O       -3.0483100000       -0.8147120000        1.2456270000
 O       -3.2896850000        1.3081350000        0.0001960000
 O       -3.0484990000       -0.8144890000       -1.2456510000

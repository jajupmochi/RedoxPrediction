%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 110-dimethyl-3f_2c_rd

 1 2
 C       -3.4931500000        0.0507610000       -0.2772690000
 C       -2.7930300000        1.2697380000       -0.2895390000
 C       -1.4117730000        1.2847010000       -0.1138440000
 C       -2.8148150000       -1.0898150000        0.0616060000
 N       -1.4710720000       -1.0975730000        0.2963170000
 C       -0.7046040000        0.0447810000        0.0502620000
 C       -0.6721880000        2.5137710000       -0.0876510000
 C        0.6721630000        2.5137790000        0.0876110000
 C        1.4117570000        1.2847150000        0.1138340000
 C        0.7046000000        0.0447910000       -0.0502680000
 H       -4.5659680000        0.0045070000       -0.4651790000
 H       -3.3223360000       -2.0441610000        0.2190030000
 H       -3.3242530000        2.2123130000       -0.4368060000
 H       -1.2257550000        3.4509190000       -0.1763910000
 H        1.2257210000        3.4509350000        0.1763370000
 C        2.7930110000        1.2697570000        0.2895650000
 C        3.4931300000        0.0507780000        0.2773480000
 N        1.4710760000       -1.0975680000       -0.2963000000
 C        2.8148130000       -1.0898060000       -0.0615330000
 H        3.3223410000       -2.0441540000       -0.2188970000
 H        4.5659400000        0.0045290000        0.4653080000
 H        3.3242300000        2.2123350000        0.4368260000
 C       -0.9814710000       -2.1737330000        1.1623660000
 H       -1.4391060000       -2.0755100000        2.1581700000
 H        0.1030140000       -2.1144780000        1.2823450000
 H       -1.2479240000       -3.1524960000        0.7404900000
 C        0.9815290000       -2.1736470000       -1.1624740000
 H       -0.1029840000       -2.1145760000       -1.2822800000
 H        1.2482260000       -3.1524330000       -0.7408070000
 H        1.4389950000       -2.0751680000       -2.1583330000


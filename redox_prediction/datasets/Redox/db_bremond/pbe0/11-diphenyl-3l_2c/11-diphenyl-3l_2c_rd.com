%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 11-diphenyl-3l_2c_rd

 1 2
 N       -0.0000000000       -0.0000000000       -3.5512800000
 C        1.1752380000        0.0380280000       -2.8519180000
 C        1.2010440000        0.0346120000       -1.4873810000
 C       -1.1752380000       -0.0380280000       -2.8519180000
 C       -1.2010440000       -0.0346120000       -1.4873810000
 C       -0.0000000000       -0.0000000000       -0.7145480000
 C        0.0000000000        0.0000000000        0.7145480000
 C        1.2010340000       -0.0349480000        1.4873810000
 C       -1.2010340000        0.0349480000        1.4873810000
 C       -1.1752270000        0.0383570000        2.8519180000
 C        1.1752270000       -0.0383570000        2.8519180000
 N        0.0000000000        0.0000000000        3.5512800000
 H        2.1804090000       -0.0847790000        1.0130180000
 H        2.0833060000       -0.1010950000        3.4518640000
 H        2.0833340000        0.1005110000       -3.4518650000
 H        2.1804330000        0.0841690000       -1.0130180000
 H       -2.0833340000       -0.1005110000       -3.4518650000
 H       -2.1804330000       -0.0841690000       -1.0130180000
 H       -2.0833060000        0.1010950000        3.4518640000
 H       -2.1804090000        0.0847790000        1.0130180000
 C        0.0000000000        0.0000000000        4.9821920000
 C        0.8425750000        0.8740390000        5.6728770000
 C        0.8417940000        0.8652040000        7.0653020000
 C        0.0000000000        0.0000000000        7.7631770000
 C       -0.8425750000       -0.8740390000        5.6728770000
 C       -0.8417940000       -0.8652040000        7.0653020000
 C       -0.0000000000       -0.0000000000       -4.9821920000
 C        0.8423330000       -0.8742730000       -5.6728770000
 C        0.8415530000       -0.8654370000       -7.0653020000
 C       -0.0000000000       -0.0000000000       -7.7631770000
 C       -0.8415530000        0.8654370000       -7.0653020000
 C       -0.8423330000        0.8742730000       -5.6728770000
 H        1.4763500000        1.5763350000        5.1266090000
 H        1.4951670000        1.5519510000        7.6076680000
 H        0.0000000000        0.0000000000        8.8551660000
 H       -1.4763500000       -1.5763350000        5.1266090000
 H       -1.4951670000       -1.5519510000        7.6076680000
 H        1.4759120000       -1.5767450000       -5.1266090000
 H        1.4947360000       -1.5523660000       -7.6076680000
 H       -0.0000000000       -0.0000000000       -8.8551660000
 H       -1.4947360000        1.5523660000       -7.6076680000
 H       -1.4759120000        1.5767450000       -5.1266090000


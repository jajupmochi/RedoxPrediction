%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 211-dimethyl-3c_2c_ox

 3 2
 C        0.0055970000        3.5023560000       -0.0630550000
 C        0.1964430000        2.8842900000        1.1886310000
 C        0.1780540000        1.4759210000        1.1760290000
 C       -0.1780540000        2.7300830000       -1.2329390000
 N       -0.1711220000        1.3968450000       -1.2066400000
 C       -0.0009740000        0.7251330000       -0.0096180000
 C        0.0009740000       -0.7251330000       -0.0096180000
 N        0.1711220000       -1.3968450000       -1.2066400000
 C       -0.4295560000        0.6190400000       -2.4382960000
 C        0.4295560000       -0.6190400000       -2.4382960000
 H       -0.1846220000        1.2431960000       -3.3092800000
 H       -1.5070580000        0.3827100000       -2.4778700000
 H        1.5070580000       -0.3827100000       -2.4778700000
 H        0.1846220000       -1.2431960000       -3.3092800000
 C       -0.1780540000       -1.4759210000        1.1760290000
 C       -0.1964430000       -2.8842900000        1.1886310000
 C       -0.0055970000       -3.5023560000       -0.0630550000
 C        0.1780540000       -2.7300830000       -1.2329390000
 H       -0.0064110000        4.5945310000       -0.1635370000
 H       -0.3413590000        3.2107700000       -2.2051290000
 H        0.0064110000       -4.5945310000       -0.1635370000
 H        0.3413590000       -3.2107700000       -2.2051290000
 H        0.3446300000        0.9506170000        2.1185680000
 H       -0.3446300000       -0.9506170000        2.1185680000
 C       -0.4045090000       -3.6702750000        2.4238660000
 H       -0.4991470000       -3.0442330000        3.3195720000
 H        0.4281360000       -4.3827940000        2.5683910000
 H       -1.3138780000       -4.2928050000        2.3280500000
 C        0.4045090000        3.6702750000        2.4238660000
 H        0.4991470000        3.0442330000        3.3195720000
 H       -0.4281360000        4.3827940000        2.5683910000
 H        1.3138780000        4.2928050000        2.3280500000

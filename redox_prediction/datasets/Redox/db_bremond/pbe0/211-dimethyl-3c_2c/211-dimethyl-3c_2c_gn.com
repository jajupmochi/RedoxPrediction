%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 211-dimethyl-3c_2c_gn

 2 1
 C       -0.0697850000        3.5055990000       -0.0883230000
 C        0.1953190000        2.8888930000        1.1460260000
 C        0.2287470000        1.4827170000        1.1584550000
 C       -0.2690880000        2.7390990000       -1.2168420000
 N       -0.2287470000        1.3914370000       -1.1662230000
 C        0.0019040000        0.7401800000        0.0080130000
 C       -0.0019040000       -0.7401800000        0.0080130000
 N        0.2287470000       -1.3914370000       -1.1662230000
 C       -0.4613360000        0.5965650000       -2.3813380000
 C        0.4613360000       -0.5965650000       -2.3813380000
 H       -0.2626640000        1.2260200000       -3.2581540000
 H       -1.5207040000        0.2917760000       -2.4092180000
 H        1.5207040000       -0.2917760000       -2.4092180000
 H        0.2626640000       -1.2260200000       -3.2581540000
 C       -0.2287470000       -1.4827170000        1.1584550000
 C       -0.1953190000       -2.8888930000        1.1460260000
 C        0.0697850000       -3.5055990000       -0.0883230000
 C        0.2690880000       -2.7390990000       -1.2168420000
 H       -0.1072420000        4.5938140000       -0.1793010000
 H       -0.4639970000        3.1830470000       -2.1959500000
 H        0.1072420000       -4.5938140000       -0.1793010000
 H        0.4639970000       -3.1830470000       -2.1959500000
 H        0.4617620000        0.9660200000        2.0902160000
 H       -0.4617620000       -0.9660200000        2.0902160000
 C       -0.4220880000       -3.6876240000        2.3780450000
 H       -0.7989780000       -3.0798830000        3.2098580000
 H        0.5274290000       -4.1540910000        2.6950120000
 H       -1.1247320000       -4.5132400000        2.1868790000
 C        0.4220880000        3.6876240000        2.3780450000
 H        0.7989780000        3.0798830000        3.2098580000
 H       -0.5274290000        4.1540910000        2.6950120000
 H        1.1247320000        4.5132400000        2.1868790000

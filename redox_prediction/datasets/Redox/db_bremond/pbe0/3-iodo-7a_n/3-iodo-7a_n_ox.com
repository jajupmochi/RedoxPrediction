%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 3-iodo-7a_n_ox

 1 2
 C        3.3416970000        4.6319210000        0.0000000000
 C        2.0855780000        5.2633950000        0.0000000000
 C        3.4311030000        3.2540940000        0.0000000000
 C        2.2627210000        2.4683380000        0.0000000000
 C        0.9981850000        3.1067110000        0.0000000000
 C        0.9257050000        4.5079710000        0.0000000000
 H        4.2540760000        5.2316180000        0.0000000000
 H        4.4072980000        2.7620390000        0.0000000000
 H        2.0209460000        6.3531900000        0.0000000000
 H       -0.0518120000        4.9964170000       -0.0000000000
 N        2.3713950000        1.1014750000        0.0000000000
 C        1.3643800000        0.1799690000        0.0000000000
 C        0.0000000000        0.5692410000        0.0000000000
 S       -0.5013840000        2.2314170000       -0.0000000000
 C        1.6799640000       -1.1942540000        0.0000000000
 C        0.6827480000       -2.1448540000        0.0000000000
 C       -0.6735400000       -1.7519140000       -0.0000000000
 C       -1.0075980000       -0.4035030000       -0.0000000000
 H        2.7282990000       -1.5043020000        0.0000000000
 H        0.9462270000       -3.2041050000        0.0000000000
 H       -2.0545260000       -0.0916530000       -0.0000000000
 H        3.3178020000        0.7302160000        0.0000000000
 I       -2.1639930000       -3.2001720000       -0.0000000000

%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 3z_2c_gn

 2 1
 C        0.0000000000       -5.6467340000       -0.8466630000
 C        0.0000000000       -4.9666530000        0.3443150000
 C        0.0000000000       -3.5519530000        0.3718570000
 N        0.0000000000       -2.8600470000       -0.8330750000
 C        0.0000000000       -4.9134410000       -2.0572490000
 C        0.0000000000       -3.5509880000       -2.0286660000
 C        0.0000000000       -2.8073790000        1.5620520000
 C       -0.0000000000       -1.4229710000        1.5642780000
 C        0.0000000000       -0.7354500000        0.3002470000
 C        0.0000000000       -1.4894520000       -0.8480940000
 C       -0.0000000000       -0.6768990000        2.7972110000
 C        0.0000000000        0.6768990000        2.7972110000
 C        0.0000000000        1.4229710000        1.5642780000
 C        0.0000000000        0.7354500000        0.3002470000
 C        0.0000000000        2.8073790000        1.5620520000
 C        0.0000000000        1.4894520000       -0.8480940000
 N        0.0000000000        2.8600470000       -0.8330750000
 C        0.0000000000        3.5519530000        0.3718570000
 C        0.0000000000        4.9666530000        0.3443150000
 C        0.0000000000        5.6467340000       -0.8466630000
 C        0.0000000000        3.5509880000       -2.0286660000
 C        0.0000000000        4.9134410000       -2.0572490000
 H        0.0000000000       -6.7388130000       -0.8645970000
 H        0.0000000000       -5.4938500000        1.3001070000
 H        0.0000000000       -5.4202830000       -3.0242850000
 H        0.0000000000       -2.9381120000       -2.9306830000
 H       -0.0000000000       -1.2300080000        3.7391040000
 H        0.0000000000        1.2300080000        3.7391040000
 H        0.0000000000       -3.3587170000        2.5042890000
 H        0.0000000000       -1.0573350000       -1.8481520000
 H        0.0000000000        1.0573350000       -1.8481520000
 H        0.0000000000        3.3587170000        2.5042890000
 H        0.0000000000        5.4938500000        1.3001070000
 H        0.0000000000        6.7388130000       -0.8645970000
 H        0.0000000000        2.9381120000       -2.9306830000
 H        0.0000000000        5.4202830000       -3.0242850000

%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 7h_n_ox

 1 2
 C       -2.0115570000       -3.9700200000        0.0000000000
 C       -0.7650220000       -4.6146940000        0.0000000000
 C       -2.0847190000       -2.5890740000        0.0000000000
 C       -0.9091100000       -1.8183460000        0.0000000000
 C        0.3432580000       -2.4697720000        0.0000000000
 C        0.4034740000       -3.8701090000        0.0000000000
 H       -2.9311960000       -4.5584520000        0.0000000000
 H       -3.0558720000       -2.0869320000        0.0000000000
 H       -0.7117380000       -5.7050380000        0.0000000000
 H        1.3763350000       -4.3677160000        0.0000000000
 N       -1.0036800000       -0.4452880000        0.0000000000
 C       -0.0000000000        0.4723210000        0.0000000000
 C        1.3517450000        0.0644810000        0.0000000000
 S        1.8453730000       -1.5948110000        0.0000000000
 C       -0.3239690000        1.8769480000        0.0000000000
 C        0.7470610000        2.8193450000        0.0000000000
 C        2.0961330000        2.3588870000        0.0000000000
 C        2.3921960000        1.0227040000        0.0000000000
 H        3.4304950000        0.6833430000        0.0000000000
 H        2.9038070000        3.0941810000        0.0000000000
 H       -1.9487460000       -0.0762760000        0.0000000000
 C       -1.6511670000        2.3623710000        0.0000000000
 C        0.4536550000        4.2000150000        0.0000000000
 C       -1.9097930000        3.7192740000        0.0000000000
 C       -0.8525870000        4.6458870000        0.0000000000
 H        1.2809290000        4.9134670000        0.0000000000
 H       -1.0672750000        5.7165550000        0.0000000000
 H       -2.5113700000        1.6875790000        0.0000000000
 H       -2.9431830000        4.0719780000        0.0000000000

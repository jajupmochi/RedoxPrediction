%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 311-dimethyl-3d_2c_rd

 1 2
 C        3.5529250000       -0.4568560000        0.0429480000
 C        2.8395240000       -1.6514200000        0.3537170000
 C        2.8030650000        0.6271430000       -0.3315110000
 N        1.4379910000        0.6012450000       -0.4028210000
 C        1.4677900000       -1.6718230000        0.3216770000
 C        0.7185280000       -0.5220250000       -0.0180460000
 C       -0.7184970000       -0.5220710000        0.0181240000
 N       -1.4379900000        0.6012210000        0.4028150000
 C       -1.4677410000       -1.6718800000       -0.3215460000
 C       -2.8394800000       -1.6514760000       -0.3536360000
 C       -3.5528990000       -0.4568960000       -0.0429990000
 C       -2.8030540000        0.6271290000        0.3314300000
 C        0.7346830000        1.7246090000       -1.0163470000
 C       -0.0000510000        2.5861640000        0.0000700000
 H        0.0275660000        1.3023930000       -1.7485850000
 H        1.4629020000        2.3169980000       -1.5849620000
 C       -0.7347830000        1.7245990000        1.0164170000
 H       -0.7009190000        3.2419320000       -0.5396890000
 H        0.7007370000        3.2420150000        0.5398160000
 H       -1.4630770000        2.3169530000        1.5849730000
 H       -0.0277010000        1.3024410000        1.7487330000
 H       -0.9288430000       -2.5709260000       -0.6229510000
 H       -3.3887740000       -2.5478440000       -0.6505150000
 H       -3.2656280000        1.5737620000        0.6177570000
 H        0.9288810000       -2.5708420000        0.6231410000
 H        3.3888400000       -2.5477590000        0.6506440000
 H        3.2656260000        1.5737600000       -0.6179040000
 C        5.0452860000       -0.3994740000        0.1147100000
 H        5.4972550000       -1.1524090000       -0.5499150000
 H        5.3967900000       -0.6178650000        1.1355750000
 H        5.4330400000        0.5865400000       -0.1741900000
 C       -5.0452520000       -0.3994970000       -0.1149360000
 H       -5.4973030000       -1.1528210000        0.5491860000
 H       -5.3966020000       -0.6172990000       -1.1359830000
 H       -5.4330610000        0.5863480000        0.1744670000

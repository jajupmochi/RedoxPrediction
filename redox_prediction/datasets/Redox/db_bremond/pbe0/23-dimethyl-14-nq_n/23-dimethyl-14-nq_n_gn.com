%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 23-dimethyl-14-nq_n_gn

 0 1
 C       -3.2799790000        0.6988060000       -0.0000150000
 C       -3.2799790000       -0.6988050000       -0.0000090000
 C       -2.0765540000        1.3981140000       -0.0000080000
 C       -0.8666050000        0.7007490000        0.0000040000
 C       -0.8666060000       -0.7007490000        0.0000130000
 C       -2.0765550000       -1.3981140000        0.0000040000
 C        0.4150750000        1.4554320000        0.0000020000
 C        1.6895060000        0.6794950000        0.0000050000
 C        1.6895060000       -0.6794950000       -0.0000100000
 C        0.4150740000       -1.4554330000        0.0000370000
 H       -4.2269240000        1.2441410000       -0.0000260000
 H       -4.2269240000       -1.2441390000       -0.0000130000
 H       -2.0443780000        2.4896530000       -0.0000170000
 H       -2.0443790000       -2.4896530000        0.0000110000
 O        0.4245460000        2.6719830000        0.0000200000
 O        0.4245440000       -2.6719830000       -0.0000710000
 C        2.9536990000        1.4758040000        0.0000300000
 H        2.7264550000        2.5487840000        0.0000470000
 H        3.5688300000        1.2412710000       -0.8839110000
 H        3.5688170000        1.2412330000        0.8839690000
 C        2.9536990000       -1.4758050000       -0.0000090000
 H        3.5691030000       -1.2408500000       -0.8836420000
 H        2.7264550000       -2.5487850000       -0.0005150000
 H        3.5685440000       -1.2416560000        0.8842380000

%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 11-bis-4-nitrophenyl-3l_2c_rd

 1 2
 N       -3.5508000000        0.0000820000       -0.0002150000
 C       -2.8510230000        1.1765650000        0.0389670000
 C       -1.4868050000        1.2011690000        0.0366580000
 C       -2.8510330000       -1.1764100000       -0.0393010000
 C       -1.4868160000       -1.2010310000       -0.0368060000
 C       -0.7149680000        0.0000640000       -0.0000210000
 C        0.7149670000        0.0000540000        0.0000770000
 C        1.4868210000        1.2011430000       -0.0368060000
 C        1.4868000000       -1.2010450000        0.0370660000
 C        2.8510170000       -1.1764410000        0.0395570000
 C        2.8510380000        1.1765210000       -0.0391100000
 N        3.5508000000        0.0000350000        0.0002710000
 H        3.4484250000       -2.0858900000        0.1061480000
 H        1.0128650000       -2.1804500000        0.0900660000
 H        1.0129060000        2.1805540000       -0.0898700000
 H        3.4484670000        2.0859620000       -0.1056190000
 H       -1.0128940000       -2.1804520000       -0.0896390000
 H       -3.4484530000       -2.0858620000       -0.1057380000
 H       -3.4484400000        2.0860250000        0.1053210000
 H       -1.0128780000        2.1805830000        0.0895570000
 C       -4.9784770000        0.0000910000       -0.0003120000
 C       -5.6688290000       -0.8744980000        0.8448830000
 C       -5.6687040000        0.8746880000       -0.8456020000
 C       -7.0584690000        0.8787380000       -0.8425640000
 C       -7.7295620000        0.0001090000       -0.0004980000
 C       -7.0585940000       -0.8785320000        0.8416550000
 H       -5.1251010000       -1.5316600000        1.5266390000
 H       -7.6344330000       -1.5390030000        1.4910770000
 H       -5.1248750000        1.5318440000       -1.5272840000
 H       -7.6342120000        1.5392160000       -1.4920640000
 C        4.9784770000        0.0000250000        0.0003680000
 C        5.6687150000        0.8747550000        0.8455100000
 C        7.0584800000        0.8787860000        0.8424700000
 C        7.7295620000        0.0000050000        0.0005530000
 C        5.6688170000       -0.8747150000       -0.8446810000
 C        7.0585810000       -0.8787670000       -0.8414530000
 H        5.1250790000       -1.5319850000       -1.5263260000
 H        7.6344110000       -1.5393550000       -1.4907650000
 H        5.1248950000        1.5320330000        1.5270810000
 H        7.6342330000        1.5393660000        1.4918590000
 N        9.2056600000       -0.0000040000        0.0006510000
 O        9.7516010000        0.7881690000        0.7350550000
 O        9.7516890000       -0.7886020000       -0.7332300000
 N       -9.2056600000        0.0001210000       -0.0005940000
 O       -9.7515890000        0.7878390000       -0.7354940000
 O       -9.7516980000       -0.7886840000        0.7330580000

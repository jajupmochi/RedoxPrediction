%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 11-bis-2-propynyl-3l_2c_ox

 3 2
 N       -3.5204800000        0.3204000000        0.1785530000
 C       -2.8783690000       -0.7850310000       -0.2457300000
 C       -2.8260700000        1.4253350000        0.5385660000
 C       -1.4942700000       -0.8172730000       -0.3188580000
 C       -0.7426220000        0.3124930000        0.0398600000
 C       -1.4461440000        1.4493140000        0.4739740000
 C        0.7426200000        0.3124920000       -0.0398600000
 C        1.4942680000       -0.8172730000        0.3188570000
 C        2.8783670000       -0.7850320000        0.2457280000
 N        3.5204780000        0.3203990000       -0.1785540000
 C        2.8260690000        1.4253340000       -0.5385660000
 C        1.4461430000        1.4493130000       -0.4739740000
 H        3.4043560000        2.2876990000       -0.8822780000
 H        0.9366330000        2.3608730000       -0.7946820000
 H        1.0256580000       -1.7319650000        0.6890650000
 H        3.4942060000       -1.6420090000        0.5295410000
 H       -3.4043570000        2.2877000000        0.8822790000
 H       -0.9366340000        2.3608730000        0.7946840000
 H       -3.4942080000       -1.6420070000       -0.5295440000
 H       -1.0256600000       -1.7319640000       -0.6890670000
 C       -5.0022420000        0.3937680000        0.2769770000
 C       -5.7390570000       -0.7759470000       -0.1190940000
 H       -5.3492850000        1.2519460000       -0.3344320000
 H       -5.2697520000        0.6400270000        1.3252030000
 C       -6.5082370000       -1.6880980000       -0.4195830000
 H       -7.1845860000       -2.4967400000       -0.6865940000
 C        5.0022410000        0.3937660000       -0.2769780000
 C        5.7390580000       -0.7759480000        0.1190930000
 H        5.3492830000        1.2519460000        0.3344310000
 H        5.2697510000        0.6400250000       -1.3252050000
 C        6.5082460000       -1.6880910000        0.4195870000
 H        7.1846050000       -2.4967250000        0.6866010000


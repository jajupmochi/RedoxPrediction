%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 11-bicyanomethyl-3l_2c_ox

 3 2
 N        3.4927980000       -0.5101830000        0.0914830000
 C        3.0004680000        0.7000210000       -0.2298960000
 C        1.6286770000        0.9291830000       -0.2214610000
 C        2.6590900000       -1.5365870000        0.4232700000
 C        1.2899120000       -1.3667070000        0.4316620000
 C        0.7310880000       -0.1125720000        0.1118510000
 C       -0.7310860000        0.1125670000        0.1118640000
 C       -1.2899030000        1.3667090000        0.4316620000
 C       -1.6286830000       -0.9291940000       -0.2214060000
 C       -3.0004750000       -0.7000300000       -0.2298180000
 C       -2.6590800000        1.5365910000        0.4232960000
 N       -3.4927970000        0.5101790000        0.0915460000
 H       -3.7125890000       -1.4877710000       -0.4959030000
 H       -1.2847740000       -1.9246240000       -0.5142420000
 H       -0.6784740000        2.2223130000        0.7285550000
 H       -3.1259560000        2.4916400000        0.6844350000
 H        0.6784900000       -2.2223060000        0.7285850000
 H        3.1259700000       -2.4916320000        0.6844180000
 H        3.7125780000        1.4877560000       -0.4960080000
 H        1.2847600000        1.9246060000       -0.5143110000
 C        4.9427840000       -0.7942640000        0.1028360000
 C       -4.9427830000        0.7942620000        0.1029320000
 C        5.7859920000        0.3183770000       -0.2536730000
 H        5.1531550000       -1.6408390000       -0.5877630000
 H        5.2342700000       -1.1547620000        1.1143740000
 N        6.4998390000        1.2018160000       -0.5369730000
 C       -5.7859960000       -0.3183670000       -0.2536060000
 H       -5.2342510000        1.1547150000        1.1144920000
 H       -5.1531640000        1.6408650000       -0.5876260000
 N       -6.4998460000       -1.2017970000       -0.5369250000


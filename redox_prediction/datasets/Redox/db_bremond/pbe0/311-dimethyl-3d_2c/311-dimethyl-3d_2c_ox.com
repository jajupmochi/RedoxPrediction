%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 311-dimethyl-3d_2c_ox

 3 2
 C        3.5505090000       -0.5090240000        0.0781110000
 C        2.8196320000       -1.6551040000        0.4575790000
 C        2.7838870000        0.6049410000       -0.3918060000
 N        1.4522580000        0.6130230000       -0.4224900000
 C        1.4371700000       -1.6464070000        0.3915450000
 C        0.7284380000       -0.4892970000       -0.0080840000
 C       -0.7284320000       -0.4893070000        0.0080850000
 N       -1.4522700000        0.6130060000        0.4224860000
 C       -1.4371460000       -1.6464230000       -0.3915500000
 C       -2.8196120000       -1.6551330000       -0.4576000000
 C       -3.5505010000       -0.5090680000       -0.0781360000
 C       -2.7838960000        0.6049080000        0.3917930000
 C        0.7421700000        1.7745280000       -1.0296550000
 C       -0.0000170000        2.6117360000       -0.0000010000
 H        0.0675000000        1.3555730000       -1.7924690000
 H        1.4858040000        2.3762610000       -1.5684880000
 C       -0.7421960000        1.7745200000        1.0296520000
 H       -0.6865910000        3.2742780000       -0.5511380000
 H        0.6865520000        3.2742830000        0.5511370000
 H       -1.4858370000        2.3762450000        1.5684840000
 H       -0.0675230000        1.3555730000        1.7924680000
 H       -0.8802990000       -2.5278860000       -0.7201880000
 H       -3.3432120000       -2.5480530000       -0.8157750000
 H       -3.2868940000        1.4985230000        0.7776050000
 H        0.8803340000       -2.5278760000        0.7201860000
 H        3.3432430000       -2.5480260000        0.8157370000
 H        3.2868750000        1.4985590000       -0.7776220000
 C        5.0148630000       -0.4424530000        0.0971170000
 H        5.4471990000       -1.0548050000        0.9061790000
 H        5.4111690000        0.5834370000        0.1184700000
 H        5.4088600000       -0.9064570000       -0.8416750000
 C       -5.0148570000       -0.4424940000       -0.0971160000
 H       -5.4473700000       -1.0560990000       -0.9050890000
 H       -5.4110710000        0.5834240000       -0.1200230000
 H       -5.4087330000       -0.9046970000        0.8426250000


%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 231213-tetramethyl-3e_2c_gn

 2 1
 C       -3.5305460000       -0.3728920000       -0.2011250000
 C       -2.8574080000       -1.3871430000        0.5218740000
 C       -1.4579820000       -1.3210130000        0.5859090000
 C       -2.7582220000        0.6127420000       -0.8000040000
 N       -1.4106920000        0.6526830000       -0.7223590000
 C       -0.7427580000       -0.3019790000       -0.0266360000
 H       -3.2214720000        1.4115780000       -1.3834130000
 C        0.7427590000       -0.3019790000        0.0266310000
 N        1.4106910000        0.6526820000        0.7223600000
 C       -0.7145080000        1.7748080000       -1.3972950000
 C       -0.5726960000        3.0039380000       -0.5037320000
 H       -1.3158680000        2.0160510000       -2.2830730000
 H        0.2520440000        1.4020760000       -1.7661550000
 C        0.7145050000        1.7748070000        1.3972940000
 C        0.5726920000        3.0039370000        0.5037310000
 H        1.3158640000        2.0160520000        2.2830720000
 H       -0.2520460000        1.4020730000        1.7661540000
 H       -1.5341830000        3.1763310000        0.0089780000
 H       -0.4390620000        3.8715220000       -1.1683110000
 H        1.5341790000        3.1763290000       -0.0089780000
 H        0.4390590000        3.8715200000        1.1683110000
 C        1.4579850000       -1.3210100000       -0.5859140000
 C        2.8574110000       -1.3871390000       -0.5218760000
 C        3.5305460000       -0.3728920000        0.2011280000
 C        2.7582190000        0.6127400000        0.8000090000
 H        3.2214690000        1.4115750000        1.3834200000
 C        3.6065600000       -2.4850310000       -1.1876620000
 H        2.9437680000       -3.1880390000       -1.7065450000
 H        4.2053040000       -3.0462770000       -0.4510790000
 H        4.3256060000       -2.0771050000       -1.9173510000
 C        5.0163070000       -0.3525360000        0.3384130000
 H        5.4999560000       -0.3073560000       -0.6503940000
 H        5.3760740000       -1.2730520000        0.8253590000
 H        5.3652990000        0.5039200000        0.9289900000
 C       -3.6065570000       -2.4850350000        1.1876600000
 H       -2.9437620000       -3.1880620000        1.7065130000
 H       -4.2053300000       -3.0462580000        0.4510850000
 H       -4.3255740000       -2.0771080000        1.9173770000
 C       -5.0163070000       -0.3525390000       -0.3384080000
 H       -5.3653030000        0.5039310000       -0.9289610000
 H       -5.4999550000       -0.3073900000        0.6504010000
 H       -5.3760690000       -1.2730420000       -0.8253810000
 H       -0.9051360000       -2.0837650000        1.1384850000
 H        0.9051410000       -2.0837610000       -1.1384930000


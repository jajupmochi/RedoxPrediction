%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 3-methyl-7a_n_gn

 0 1
 C       -4.0065000000        0.7947070000        0.4468460000
 C       -4.0250730000       -0.5925770000        0.5645800000
 C       -2.8316010000        1.4526940000        0.0900230000
 C       -1.6634600000        0.7340830000       -0.1901980000
 C       -1.6918580000       -0.6689240000       -0.0867390000
 C       -2.8605870000       -1.3169610000        0.3089450000
 H       -4.9087270000        1.3775460000        0.6460900000
 H       -2.8141800000        2.5441970000        0.0190050000
 H       -4.9391310000       -1.1140720000        0.8555690000
 H       -2.8556390000       -2.4060470000        0.4001220000
 N       -0.4960200000        1.3917180000       -0.5583970000
 C        0.7792560000        0.9022980000       -0.2903430000
 C        1.0126740000       -0.4815330000       -0.2001150000
 S       -0.2852550000       -1.6240580000       -0.5785150000
 C        1.8575350000        1.7685120000       -0.0961810000
 C        3.1360130000        1.2751970000        0.1593810000
 C        3.3762600000       -0.0958900000        0.2653490000
 C        2.2857690000       -0.9601110000        0.0944860000
 H        1.6919410000        2.8484110000       -0.1522750000
 H        3.9607910000        1.9803670000        0.2922750000
 H        2.4314850000       -2.0411940000        0.1764670000
 H       -0.5670350000        2.3999820000       -0.5953520000
 C        4.7475200000       -0.6409110000        0.5452590000
 H        5.4631060000        0.1655730000        0.7590950000
 H        5.1366220000       -1.2120610000       -0.3136490000
 H        4.7413080000       -1.3232990000        1.4099170000


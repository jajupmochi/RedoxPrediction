%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 1122-tetramethyl-3p_2c_gn

 2 1
 N        3.4821690000        0.5008220000       -0.1243560000
 C        2.9963330000       -0.7022290000        0.2881140000
 C        1.6137750000       -0.8959960000        0.2812410000
 C        2.6546390000        1.4896190000       -0.5265640000
 C        1.2851710000        1.3323390000       -0.5356380000
 C        0.7339700000        0.1099010000       -0.1253070000
 H        3.1288430000        2.4183510000       -0.8487920000
 H        0.6677620000        2.1587260000       -0.8916240000
 H        1.2396730000       -1.8592230000        0.6323710000
 C       -0.7339720000       -0.1099010000       -0.1252980000
 C       -1.2851750000       -1.3323490000       -0.5356000000
 C       -2.6546430000       -1.4896300000       -0.5265100000
 C       -1.6137740000        0.8960040000        0.2812340000
 C       -2.9963310000        0.7022370000        0.2881230000
 N       -3.4821700000       -0.5008250000       -0.1243140000
 H       -1.2396690000        1.8592400000        0.6323360000
 H       -0.6677680000       -2.1587420000       -0.8915750000
 H       -3.1288490000       -2.4183680000       -0.8487140000
 C       -4.9321190000       -0.7430860000       -0.1380630000
 H       -5.3374720000       -0.6362820000        0.8769030000
 H       -5.1222440000       -1.7594650000       -0.4977690000
 H       -5.4239760000       -0.0242630000       -0.8068330000
 C        4.9321190000        0.7430820000       -0.1381230000
 H        5.4239790000        0.0241850000       -0.8068110000
 H        5.3374650000        0.6363910000        0.8768570000
 H        5.1222450000        1.7594210000       -0.4979420000
 C       -3.9408170000        1.7611970000        0.7319920000
 H       -3.3930400000        2.6619830000        1.0313090000
 H       -4.5443730000        1.4275550000        1.5916310000
 H       -4.6422130000        2.0382840000       -0.0715360000
 C        3.9408220000       -1.7611850000        0.7319850000
 H        4.5444950000       -1.4274940000        1.5915200000
 H        4.6421110000       -2.0383790000       -0.0716020000
 H        3.3930400000       -2.6619210000        1.0314450000


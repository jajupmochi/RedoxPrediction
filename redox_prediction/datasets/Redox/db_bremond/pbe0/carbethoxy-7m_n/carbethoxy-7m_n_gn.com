%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 carbethoxy-7m_n_gn

 0 1
 C        1.1489480000        3.0636250000       -1.4826460000
 C       -0.0094100000        3.7047090000       -1.0378890000
 C        1.3691300000        1.7141490000       -1.2234490000
 C        0.4372820000        0.9840110000       -0.4794160000
 C       -0.7137640000        1.6452140000       -0.0356520000
 C       -0.9560940000        2.9862090000       -0.3165120000
 H        1.8877470000        3.6195040000       -2.0647480000
 H        2.2627450000        1.2184990000       -1.6017400000
 H       -0.1764250000        4.7601410000       -1.2628770000
 H       -1.8858150000        3.4390650000        0.0353250000
 N        0.5759300000       -0.4127060000       -0.2522250000
 C       -0.5675470000       -1.2424750000       -0.4172760000
 C       -1.8148090000       -0.7888950000        0.0286230000
 S       -1.8424480000        0.7151340000        0.9636180000
 C       -0.5053110000       -2.4785170000       -1.0662740000
 C       -1.6690300000       -3.2206240000       -1.2506580000
 C       -2.9085340000       -2.7438470000       -0.8198260000
 C       -2.9848790000       -1.5121940000       -0.1788170000
 H        0.4539120000       -2.8547290000       -1.4181990000
 H       -1.6058150000       -4.1854040000       -1.7593850000
 H       -3.9325820000       -1.0952540000        0.1692830000
 H       -3.8135770000       -3.3318360000       -0.9861770000
 O       -3.1712670000        1.3165410000        0.8967250000
 O       -1.2083140000        0.4641320000        2.2539710000
 C        1.7391590000       -0.9844120000        0.2665810000
 O        2.6521920000       -0.0562650000        0.5422660000
 O        1.8811160000       -2.1652220000        0.4502380000
 C        3.8759460000       -0.5161750000        1.1240670000
 C        4.8557870000       -0.9917300000        0.0768650000
 H        4.2609390000        0.3546700000        1.6720070000
 H        3.6459520000       -1.3163150000        1.8424340000
 H        4.4664170000       -1.8754110000       -0.4484240000
 H        5.8053740000       -1.2730870000        0.5566350000
 H        5.0677260000       -0.2008240000       -0.6583910000


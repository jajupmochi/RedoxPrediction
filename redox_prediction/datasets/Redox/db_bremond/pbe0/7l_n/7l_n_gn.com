%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 7l_n_gn

 0 1
 C       -3.7044340000        1.2214630000        0.1958910000
 C       -4.0146140000       -0.0872440000       -0.1739280000
 C       -2.3788560000        1.6153000000        0.3132130000
 C       -1.3121200000        0.7325410000        0.0576230000
 C       -1.6428720000       -0.5848780000       -0.3222570000
 C       -2.9770880000       -0.9803510000       -0.4142420000
 H       -4.4982150000        1.9441120000        0.3977170000
 H       -2.1448630000        2.6431050000        0.6076790000
 H       -5.0529610000       -0.4108680000       -0.2685100000
 H       -3.1664290000       -2.0167770000       -0.7012450000
 N       -0.0273240000        1.2580930000        0.1649850000
 C        1.2617570000        0.7755750000        0.0460080000
 C        1.6383020000       -0.5821070000        0.1516780000
 S        0.4741770000       -1.8136560000        0.5660440000
 C        2.2865560000        1.7208070000       -0.1632470000
 C        3.6167130000        1.3405200000       -0.2587500000
 C        3.9778840000       -0.0066840000       -0.1801810000
 C        2.9811280000       -0.9555950000        0.0035580000
 H        2.0182100000        2.7783270000       -0.2467810000
 H        4.3802880000        2.1066560000       -0.4133480000
 H        3.2250060000       -2.0198070000        0.0508510000
 H        5.0213630000       -0.3122030000       -0.2766140000
 H       -0.0468740000        2.2651760000        0.2654750000
 O       -0.6906540000       -1.5027450000       -0.6448780000

%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 11-bis-2-hydroxyethyl-3l_2c_gn

 2 1
 N        3.5153740000        0.1684760000       -0.3477890000
 C        2.8962230000       -0.9657720000       -0.7279030000
 C        2.7986220000        1.2348270000        0.0565450000
 C        1.5160090000       -1.0634540000       -0.7181000000
 C        0.7412690000        0.0335230000       -0.3142290000
 C        1.4156450000        1.1983830000        0.0793650000
 C       -0.7412700000       -0.0335120000       -0.3142300000
 C       -1.4156450000       -1.1983750000        0.0793590000
 C       -2.7986220000       -1.2348190000        0.0565380000
 N       -3.5153740000       -0.1684650000       -0.3477910000
 C       -2.8962230000        0.9657850000       -0.7279000000
 C       -1.5160100000        1.0634660000       -0.7180960000
 H       -3.5371350000        1.7880930000       -1.0536880000
 H       -1.0616040000        1.9942390000       -1.0630460000
 H       -0.8810000000       -2.0846150000        0.4264700000
 H       -3.3641900000       -2.1202020000        0.3553870000
 H        3.3641900000        2.1202090000        0.3553990000
 H        0.8810000000        2.0846220000        0.4264790000
 H        3.5371350000       -1.7880790000       -1.0536950000
 H        1.0616040000       -1.9942250000       -1.0630540000
 C        4.9882270000        0.2284580000       -0.3331100000
 C        5.5794140000       -0.2278560000        1.0002340000
 H        5.2945590000        1.2621710000       -0.5419640000
 H        5.3683080000       -0.4055530000       -1.1453210000
 O        6.9426720000       -0.1032400000        0.8199280000
 H        5.2537200000       -1.2685670000        1.2091990000
 H        5.1804670000        0.4103230000        1.8164770000
 C       -4.9882270000       -0.2284490000       -0.3331130000
 C       -5.5794130000        0.2278350000        1.0002410000
 H       -5.2945580000       -1.2621590000       -0.5419870000
 H       -5.3683090000        0.4055770000       -1.1453110000
 O       -6.9426700000        0.1032000000        0.8199420000
 H       -5.2537340000        1.2685460000        1.2092220000
 H       -5.1804530000       -0.4103520000        1.8164720000
 H        7.4202640000       -0.3670480000        1.6147270000
 H       -7.4202650000        0.3670260000        1.6147340000


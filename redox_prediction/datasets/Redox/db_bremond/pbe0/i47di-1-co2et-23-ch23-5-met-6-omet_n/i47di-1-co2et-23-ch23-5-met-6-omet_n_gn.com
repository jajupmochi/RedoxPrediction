%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 i47di-1-co2et-23-ch23-5-met-6-omet_n_gn

 0 1
 C        2.7450770000       -0.9255690000       -0.1530830000
 C        3.1629500000        0.3706060000       -0.2045400000
 C        1.2903720000       -1.3671710000       -0.1218500000
 C        2.1705590000        1.4753430000       -0.1418900000
 C        0.3144930000       -0.2672150000       -0.0241460000
 C        0.7681930000        1.0780140000       -0.0456830000
 C       -1.0843350000       -0.2417910000        0.0594810000
 C       -0.3519150000        1.8873670000        0.0278190000
 N       -1.4429730000        1.0874550000        0.0913210000
 O        1.0239580000       -2.5432270000       -0.1811370000
 O        2.5094660000        2.6483570000       -0.1840860000
 C       -2.0616080000       -1.3428180000        0.1281840000
 O       -3.3112510000       -0.8967660000       -0.1206070000
 O       -1.8121630000       -2.4908840000        0.3718300000
 C       -4.3393060000       -1.8849410000       -0.0759110000
 C       -5.6533360000       -1.2161700000       -0.3883120000
 H       -4.3413410000       -2.3569660000        0.9199880000
 H       -4.0998070000       -2.6806080000       -0.7993570000
 H       -5.8917740000       -0.4377680000        0.3520620000
 H       -6.4643280000       -1.9589110000       -0.3738530000
 H       -5.6333040000       -0.7512070000       -1.3852010000
 C       -2.6915630000        1.8383500000        0.1489840000
 C       -2.2045140000        3.2529000000        0.5033640000
 H       -3.3737290000        1.4083510000        0.8920150000
 H       -3.1962100000        1.7876960000       -0.8291260000
 C       -0.7418680000        3.3227060000        0.0230140000
 H       -2.2382800000        3.3856420000        1.5956730000
 H       -2.8424450000        4.0290010000        0.0589820000
 H       -0.0904360000        3.9371270000        0.6590190000
 H       -0.6563140000        3.7315790000       -0.9989090000
 C        4.5897940000        0.7807170000       -0.3975120000
 H        5.1003670000        0.1186970000       -1.1114190000
 H        5.1639910000        0.7733650000        0.5431980000
 H        4.6109560000        1.8117390000       -0.7748070000
 O        3.5594890000       -1.9883490000       -0.1814830000
 C        4.6608480000       -2.0743690000        0.6926010000
 H        5.5808780000       -1.6574410000        0.2541960000
 H        4.8203920000       -3.1426170000        0.8946670000
 H        4.4531520000       -1.5586670000        1.6443710000


%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 37-dimethoxy-7a_n_rd

-1 2
 C        3.6775390000        1.1314610000        0.0714510000
 C        3.7834540000       -0.2718900000        0.1396990000
 C        2.3902320000        1.6955150000       -0.0856370000
 C        1.2449370000        0.9217380000       -0.2029310000
 C        1.3681400000       -0.5033070000       -0.1557180000
 C        2.6516550000       -1.0694010000        0.0420110000
 H        4.5464320000        1.7851260000        0.1415410000
 H        2.2896320000        2.7873730000       -0.1107100000
 H        2.7672320000       -2.1534470000        0.1206250000
 N       -0.0000000000        1.5190520000       -0.3751840000
 C       -1.2449370000        0.9217370000       -0.2029310000
 C       -1.3681400000       -0.5033080000       -0.1557190000
 S       -0.0000000000       -1.5048210000       -0.6071860000
 C       -2.3902320000        1.6955150000       -0.0856360000
 C       -3.6775390000        1.1314610000        0.0714520000
 C       -3.7834540000       -0.2718900000        0.1396990000
 C       -2.6516560000       -1.0694010000        0.0420100000
 H       -2.2896320000        2.7873730000       -0.1107090000
 H       -4.5464320000        1.7851260000        0.1415430000
 H       -2.7672330000       -2.1534470000        0.1206240000
 H       -0.0000000000        2.5286160000       -0.3540170000
 O       -4.9784280000       -0.9323620000        0.3274540000
 C       -6.1209460000       -0.1698820000        0.5072580000
 H       -6.0504400000        0.5031100000        1.3847400000
 H       -6.9565900000       -0.8680870000        0.6701680000
 H       -6.3594900000        0.4570080000       -0.3751620000
 O        4.9784290000       -0.9323620000        0.3274550000
 C        6.1209460000       -0.1698830000        0.5072590000
 H        6.3594910000        0.4570070000       -0.3751620000
 H        6.9565900000       -0.8680880000        0.6701690000
 H        6.0504410000        0.5031100000        1.3847400000


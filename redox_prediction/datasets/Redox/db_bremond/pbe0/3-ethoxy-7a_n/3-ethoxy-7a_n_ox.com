%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 3-ethoxy-7a_n_ox

 1 2
 C       -4.8064210000        0.9078900000        0.1228690000
 C       -4.8860540000       -0.4937620000        0.1632970000
 C       -3.5775240000        1.5329890000        0.0448890000
 C       -2.3946220000        0.7702050000        0.0053200000
 C       -2.4765570000       -0.6409380000        0.0460430000
 C       -3.7323350000       -1.2594580000        0.1252100000
 H       -5.7173820000        1.5087420000        0.1529700000
 H       -3.5116590000        2.6238270000        0.0131300000
 H       -5.8588200000       -0.9853640000        0.2248860000
 H       -3.7927760000       -2.3500770000        0.1564890000
 N       -1.1838450000        1.4117570000       -0.0721640000
 C        0.0597440000        0.8530990000       -0.1215190000
 C        0.2506360000       -0.5573500000       -0.0943740000
 S       -1.0780620000       -1.6757290000        0.0026230000
 C        1.1950880000        1.6843540000       -0.2008470000
 C        2.4660000000        1.1573250000       -0.2519320000
 C        2.6556010000       -0.2489980000       -0.2277990000
 C        1.5330080000       -1.0863900000       -0.1467160000
 H        1.0601290000        2.7690790000       -0.2200530000
 H        3.3180880000        1.8341730000       -0.3090890000
 H        1.6976130000       -2.1656850000       -0.1295610000
 H       -1.2200540000        2.4270780000       -0.0969100000
 O        3.8267770000       -0.8529200000       -0.2865320000
 C        5.0599950000       -0.1293790000       -0.3520310000
 C        5.5265990000        0.3277060000        1.0105200000
 H        4.9627780000        0.7035580000       -1.0678070000
 H        5.7663130000       -0.8492960000       -0.7861860000
 H        6.5151640000        0.8013850000        0.9198950000
 H        4.8426210000        1.0595980000        1.4659330000
 H        5.6207150000       -0.5280450000        1.6941600000

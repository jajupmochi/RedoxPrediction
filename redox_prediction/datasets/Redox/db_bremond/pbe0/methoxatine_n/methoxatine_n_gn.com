%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 methoxatine_n_gn

 0 1
 C        0.5958990000        2.5593240000        0.0495470000
 C       -0.9287340000        2.7499950000        0.1796720000
 C        1.1558130000        1.1627070000       -0.0081550000
 C        0.3338730000       -0.0002020000        0.0268840000
 C       -1.1024550000        0.2419290000        0.0125900000
 C       -1.6969050000        1.5173630000        0.1075620000
 O       -1.3895430000        3.8593120000        0.2977770000
 O        1.3173000000        3.5223340000        0.0047380000
 C        0.9949370000       -1.2565280000        0.1010540000
 C        2.3935800000       -1.3098980000        0.0020070000
 C        2.5400910000        1.0718210000       -0.0727030000
 C        3.1781450000       -0.1672160000       -0.1219190000
 H        2.8774420000       -2.2915610000        0.0277040000
 H        3.1265820000        1.9934630000       -0.0941880000
 C        4.6742750000       -0.2062620000       -0.2140520000
 C        0.3058040000       -2.5776660000        0.2830910000
 C       -3.0933670000        1.3483110000        0.0590090000
 C       -3.3136230000       -0.0122040000       -0.0797180000
 N       -2.1083280000       -0.6544310000       -0.1039840000
 C       -4.5952970000       -0.7184300000       -0.2040030000
 O       -5.6727630000       -0.1848350000       -0.1839780000
 O       -4.4193720000       -2.0412110000       -0.3444490000
 H       -5.2998790000       -2.4385380000       -0.4216850000
 O       -0.7377570000       -2.9007740000       -0.2261820000
 O        0.9532910000       -3.4568890000        1.0511960000
 H        1.6800250000       -3.0302870000        1.5229340000
 O        5.3642990000        0.7030540000        0.1435530000
 O        5.2092520000       -1.3336750000       -0.7113480000
 H        4.5334950000       -1.8742830000       -1.1374720000
 H       -3.8532950000        2.1241950000        0.1051050000
 H       -1.9559460000       -1.6587760000       -0.2501620000

%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 37-dimethoxy-7a_n_gn

 0 1
 C        1.1203240000        0.1995520000        3.5888910000
 C       -0.2701850000        0.2339890000        3.7049730000
 C        1.7057430000       -0.0812690000        2.3516570000
 C        0.9372310000       -0.3682380000        1.2243530000
 C       -0.4650650000       -0.3427020000        1.3564070000
 C       -1.0548850000       -0.0262400000        2.5727590000
 H        1.7649060000        0.4013770000        4.4444590000
 H        2.7964520000       -0.0858520000        2.2679010000
 H       -2.1416540000        0.0088480000        2.6711380000
 N        1.5275060000       -0.6752640000       -0.0000000000
 C        0.9372310000       -0.3682380000       -1.2243530000
 C       -0.4650650000       -0.3427020000       -1.3564070000
 S       -1.4923850000       -0.8275010000        0.0000000000
 C        1.7057430000       -0.0812690000       -2.3516570000
 C        1.1203240000        0.1995520000       -3.5888910000
 C       -0.2701850000        0.2339890000       -3.7049730000
 C       -1.0548850000       -0.0262400000       -2.5727590000
 H        2.7964520000       -0.0858520000       -2.2679010000
 H        1.7649060000        0.4013770000       -4.4444590000
 H       -2.1416540000        0.0088480000       -2.6711380000
 H        2.5384860000       -0.6603020000       -0.0000000000
 O       -0.9441370000        0.5066170000       -4.8451650000
 C       -0.2141920000        0.7794220000       -6.0053200000
 H        0.4266420000        1.6726350000       -5.8907510000
 H       -0.9449890000        0.9703500000       -6.8019070000
 H        0.4214720000       -0.0738000000       -6.3045640000
 O       -0.9441370000        0.5066170000        4.8451650000
 C       -0.2141920000        0.7794220000        6.0053200000
 H        0.4214720000       -0.0738000000        6.3045640000
 H       -0.9449890000        0.9703500000        6.8019070000
 H        0.4266420000        1.6726350000        5.8907510000

%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 11-bicyanomethyl-3l_2c_gn

 2 1
 N       -3.4772520000        0.5296620000        0.1160260000
 C       -2.9852740000       -0.6573030000       -0.2843100000
 C       -1.6194160000       -0.8925030000       -0.2866530000
 C       -2.6510390000        1.5189210000        0.5199700000
 C       -1.2817010000        1.3399980000        0.5292120000
 C       -0.7340560000        0.1118960000        0.1220080000
 C        0.7340580000       -0.1119090000        0.1219840000
 C        1.2817190000       -1.3399960000        0.5292120000
 C        1.6194040000        0.8924760000       -0.2867440000
 C        2.9852620000        0.6572790000       -0.2844400000
 C        2.6510570000       -1.5189170000        0.5199280000
 N        3.4772550000       -0.5296720000        0.1159200000
 H        3.7102600000        1.4111240000       -0.6046160000
 H        1.2698200000        1.8648440000       -0.6398130000
 H        0.6569730000       -2.1626230000        0.8832700000
 H        3.1179880000       -2.4543630000        0.8384450000
 H       -0.6569410000        2.1626380000        0.8832170000
 H       -3.1179570000        2.4543790000        0.8384690000
 H       -3.7102840000       -1.4111570000       -0.6044410000
 H       -1.2698470000       -1.8648830000       -0.6397010000
 C       -4.9425730000        0.8092140000        0.1312940000
 C        4.9425760000       -0.8092290000        0.1311220000
 C       -5.7276150000       -0.3336350000       -0.2985810000
 H       -5.1344710000        1.6674880000       -0.5334110000
 H       -5.2257950000        1.0933740000        1.1579250000
 N       -6.3319430000       -1.2562460000       -0.6442260000
 C        5.7276100000        0.3336640000       -0.2986480000
 H        5.2258210000       -1.0935030000        1.1577150000
 H        5.1344560000       -1.6674300000       -0.5336820000
 N        6.3319270000        1.2563090000       -0.6442220000

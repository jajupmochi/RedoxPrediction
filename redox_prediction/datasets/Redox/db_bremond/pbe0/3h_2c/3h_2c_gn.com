%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 3h_2c_gn

 2 1
 C        3.5643950000       -0.3135060000       -0.0778190000
 C        2.8337260000       -1.4488510000        0.1762030000
 C        2.8625130000        0.8547220000       -0.3505250000
 C        1.4190830000       -1.4110050000        0.1718670000
 N        1.5254180000        0.9217610000       -0.3233170000
 C        0.7349940000       -0.1723790000       -0.0218620000
 C        0.7163130000       -2.6429330000        0.3000520000
 C       -0.6283670000       -2.6692950000        0.1374360000
 C       -1.3607250000       -1.4567090000       -0.0243610000
 C       -0.7236290000       -0.1753080000        0.0193280000
 C       -2.7562340000       -1.5572140000       -0.2263070000
 C       -3.5209110000       -0.4235570000       -0.3619090000
 C       -2.8896680000        0.7943040000       -0.1564720000
 N       -1.5714990000        0.9277770000        0.0686080000
 H        4.6558990000       -0.3075250000       -0.0975680000
 H        3.3825590000        1.7795780000       -0.6075380000
 H        3.3329440000       -2.4038010000        0.3611790000
 H        1.2854250000       -3.5602440000        0.4641140000
 H       -1.1785800000       -3.6126100000        0.1417180000
 C       -1.2574420000        2.3041620000        0.5834820000
 C        0.1870300000        2.7177730000        0.6209050000
 H       -1.7069300000        2.3646320000        1.5861620000
 H       -1.8138630000        2.9845460000       -0.0751470000
 C        0.9343110000        2.2395440000       -0.6036850000
 H        0.6993810000        2.3720700000        1.5321010000
 H        0.2021770000        3.8171070000        0.6687960000
 H        1.7573310000        2.9119410000       -0.8686110000
 H        0.2903900000        2.1536880000       -1.4943370000
 H       -4.5953200000       -0.4534420000       -0.5528350000
 H       -3.4700450000        1.7187960000       -0.1125930000
 H       -3.2111310000       -2.5499870000       -0.2804840000


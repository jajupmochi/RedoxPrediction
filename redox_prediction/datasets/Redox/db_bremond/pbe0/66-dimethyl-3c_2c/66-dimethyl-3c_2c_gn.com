%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 66-dimethyl-3c_2c_gn

 2 1
 C       -3.7010400000       -0.1000260000       -0.0709400000
 C       -3.2628030000       -1.3619820000        0.3235900000
 C       -1.8956090000       -1.6399920000        0.3409500000
 C       -2.7579190000        0.8506450000       -0.4236060000
 N       -1.4423190000        0.5624950000       -0.4045380000
 C       -0.9808150000       -0.6638760000       -0.0433550000
 C        0.4761030000       -0.9046400000       -0.0774180000
 N        1.3360580000        0.1531550000       -0.0017020000
 C       -0.4555110000        1.5755860000       -0.7966090000
 C        0.7665970000        1.5546410000        0.1193380000
 H       -0.9345660000        2.5607490000       -0.7434730000
 H       -0.1659580000        1.3918570000       -1.8449010000
 C        0.9880030000       -2.1947870000       -0.1781710000
 C        2.3659030000       -2.4104430000       -0.1645440000
 C        3.2165090000       -1.3168820000       -0.0517080000
 C        2.6661380000       -0.0461910000        0.0221260000
 H       -4.7616440000        0.1599730000       -0.0969940000
 H        4.3027330000       -1.4298100000       -0.0327310000
 H       -1.5451990000       -2.6161320000        0.6776440000
 H        0.3088870000       -3.0404500000       -0.2907050000
 H        2.7642760000       -3.4251740000       -0.2465890000
 H       -3.9798260000       -2.1299070000        0.6263060000
 H       -3.0310990000        1.8630140000       -0.7305650000
 H        3.2934430000        0.8411500000        0.0975030000
 C        1.7453670000        2.6022860000       -0.4012370000
 H        2.1080750000        2.3762280000       -1.4151680000
 H        2.6005160000        2.7455040000        0.2733620000
 H        1.2281870000        3.5718160000       -0.4395260000
 C        0.4190730000        1.8020630000        1.5869470000
 H       -0.2991670000        1.0735570000        1.9916870000
 H       -0.0108270000        2.8082500000        1.6984440000
 H        1.3260250000        1.7614100000        2.2072160000


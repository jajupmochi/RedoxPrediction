%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 1t-nch32_n_ox

 1 2
 C        1.7919560000       -3.6858150000       -0.1026490000
 C        3.1813090000       -3.6012600000       -0.2901300000
 C        1.0265730000       -2.5332820000        0.0457940000
 C        3.8055430000       -2.3669780000       -0.3221890000
 C        3.0525650000       -1.1730380000       -0.1734340000
 C        1.6327970000       -1.2700370000        0.0012220000
 H        1.3071230000       -4.6638260000       -0.0664530000
 H       -0.0502770000       -2.6082860000        0.2128200000
 H        3.7700910000       -4.5140660000       -0.4038320000
 H        4.8837430000       -2.2663190000       -0.4594360000
 C        0.8803890000       -0.0453610000        0.1199120000
 C        1.6074080000        1.2018640000        0.1007090000
 C        3.0309860000        1.1414370000       -0.0697520000
 N        3.7159520000       -0.0067980000       -0.2015460000
 C        0.9831960000        2.4522340000        0.2266970000
 C        1.7317850000        3.6236890000        0.1794190000
 C        3.1245080000        3.5734490000        0.0092750000
 C        3.7660630000        2.3538220000       -0.1135470000
 H        4.8466430000        2.2789700000       -0.2486370000
 H        3.7013790000        4.5002160000       -0.0241470000
 H       -0.0981950000        2.5121030000        0.3681980000
 H        1.2303020000        4.5887440000        0.2789200000
 N       -0.4194630000       -0.0830320000        0.3209400000
 C       -1.5380710000        0.1167800000       -0.2669010000
 C       -2.8179060000       -0.0124300000        0.4858990000
 C       -3.9701410000       -0.0617360000       -0.2558550000
 C       -4.0088030000        0.1119370000       -1.7010230000
 C       -2.7135270000        0.3567350000       -2.3932420000
 C       -1.5604460000        0.3798670000       -1.7096320000
 H       -4.9442540000       -0.1956500000        0.2133850000
 H       -2.7593900000        0.5167610000       -3.4735170000
 H       -0.5960440000        0.5438020000       -2.1968810000
 O       -5.0436520000        0.0976870000       -2.3401630000
 N       -2.7812660000       -0.2235200000        1.8341780000
 C       -1.8520720000        0.4329910000        2.7351040000
 H       -1.2720370000       -0.2993640000        3.3195270000
 H       -2.4093240000        1.0719720000        3.4412640000
 H       -1.1492700000        1.0756250000        2.1959160000
 C       -3.9677990000       -0.7397360000        2.4792740000
 H       -3.6963020000       -1.1122220000        3.4763320000
 H       -4.3857450000       -1.5763070000        1.9012640000
 H       -4.7536590000        0.0290180000        2.6018710000

%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 3-methoxy-3d_2c_gn

 2 1
 C       -4.1535780000       -0.8917070000       -0.0136390000
 C       -3.3600680000       -1.8973530000       -0.5583330000
 C       -3.5388400000        0.2350680000        0.5067970000
 N       -2.1981900000        0.3713670000        0.4949730000
 C       -1.9730170000       -1.7381620000       -0.5763300000
 C       -1.3936360000       -0.5899350000       -0.0477270000
 C        0.0782760000       -0.4199160000       -0.0235770000
 N        0.6474110000        0.7228370000       -0.5122140000
 C        0.9172210000       -1.4254340000        0.4490320000
 C        2.2998290000       -1.2788900000        0.4441970000
 C        2.8592940000       -0.0885310000       -0.0506350000
 C        1.9702610000        0.8978570000       -0.5292400000
 C       -1.5777940000        1.5709670000        1.0943810000
 C       -0.9992040000        2.5101810000        0.0430280000
 H       -0.8019880000        1.2129570000        1.7879600000
 H       -2.3373170000        2.0784450000        1.7019350000
 C       -0.2300800000        1.7839550000       -1.0539190000
 H       -0.3545690000        3.2347740000        0.5643210000
 H       -1.7938320000        3.0963360000       -0.4444330000
 H        0.3929770000        2.4809250000       -1.6275000000
 H       -0.9073060000        1.2974880000       -1.7719910000
 H        0.4745420000       -2.3401610000        0.8489930000
 H        2.9306160000       -2.0831020000        0.8274480000
 H        2.3620270000        1.8301530000       -0.9417710000
 H       -3.8141670000       -2.7993490000       -0.9774120000
 H       -5.2425110000       -0.9699560000        0.0170500000
 H       -1.3326800000       -2.5012360000       -1.0230860000
 H       -4.1061570000        1.0521600000        0.9582290000
 O        4.1248030000        0.2135930000       -0.1250820000
 C        5.1329020000       -0.7035120000        0.3102550000
 H        6.0883230000       -0.1993260000        0.1323910000
 H        5.0886220000       -1.6306500000       -0.2809290000
 H        5.0210520000       -0.9151650000        1.3844140000


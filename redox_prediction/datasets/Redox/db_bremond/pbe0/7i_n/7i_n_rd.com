%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 7i_n_rd

-1 2
 C       -4.7163620000        0.2810730000       -0.4332170000
 C       -3.6646310000        1.1591230000       -0.1850230000
 C       -2.3635800000        0.6933760000        0.0800120000
 C       -2.1418060000       -0.7112380000        0.0853420000
 C       -4.4904530000       -1.0985170000       -0.4450070000
 C       -3.2005810000       -1.5746180000       -0.1925530000
 N       -1.3414170000        1.5698310000        0.3460220000
 C        0.0307080000        1.3056800000        0.2274290000
 C        0.5132490000       -0.0245090000        0.2031720000
 S       -0.5901410000       -1.3597420000        0.5992990000
 C        0.9199350000        2.3653030000        0.0878180000
 C        2.2942710000        2.1350460000       -0.0691890000
 C        2.8148030000        0.8156330000       -0.1155750000
 C        1.9009980000       -0.3014820000        0.0146340000
 C        4.1945820000        0.5433260000       -0.2801680000
 C        2.4382510000       -1.6103110000       -0.0351650000
 C        4.6901680000       -0.7649990000       -0.3257120000
 C        3.8108100000       -1.8380950000       -0.2054910000
 H       -5.7148090000        0.6811900000       -0.6314190000
 H       -3.8427050000        2.2400190000       -0.1888710000
 H       -5.3033110000       -1.7981110000       -0.6545620000
 H       -3.0006090000       -2.6503720000       -0.1971750000
 H        0.5292710000        3.3897040000        0.0793570000
 H        2.9816590000        2.9793730000       -0.1720840000
 H        4.8798990000        1.3930710000       -0.3686260000
 H        5.7636450000       -0.9392400000       -0.4519570000
 H        1.7562900000       -2.4603820000        0.0531180000
 H        4.1857020000       -2.8660680000       -0.2411220000
 H       -1.5850370000        2.5491340000        0.2945590000

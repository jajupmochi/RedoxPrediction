%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 6-phenyl-3c_2c_ox

 3 2
 C       -3.7409580000       -2.4140420000        0.0353130000
 C       -4.3587700000       -1.1940540000       -0.2312740000
 C       -3.5924650000       -0.0249470000       -0.2516740000
 C       -2.3737060000       -2.4338610000        0.2594510000
 N       -1.6494220000       -1.2945560000        0.2354990000
 C       -2.2255600000       -0.0806460000        0.0015530000
 C       -1.3710560000        1.1243650000        0.0221150000
 N       -0.0218070000        1.0038990000       -0.1863830000
 C       -0.2113370000       -1.3342500000        0.5170940000
 C        0.5364180000       -0.3675820000       -0.3980670000
 H        0.1494040000       -2.3566630000        0.3447240000
 H       -0.0564340000       -1.0902930000        1.5820000000
 C       -1.9116190000        2.3934370000        0.2142220000
 C       -1.1037660000        3.5290610000        0.1458090000
 C        0.2550400000        3.3773030000       -0.1241510000
 C        0.7637290000        2.1001730000       -0.2819310000
 H       -4.3029060000       -3.3517370000        0.0607900000
 H       -1.8296750000       -3.3598360000        0.4641050000
 H        0.9248820000        4.2372340000       -0.2077100000
 H        1.8194360000        1.9308330000       -0.4977290000
 H       -4.0704880000        0.9267510000       -0.4887450000
 H       -2.9749100000        2.5015430000        0.4340940000
 H       -5.4338580000       -1.1458870000       -0.4303330000
 H       -1.5373770000        4.5223350000        0.2982550000
 C        2.0422160000       -0.4807540000       -0.2057030000
 H        0.3120390000       -0.6287410000       -1.4464990000
 C        2.8487960000       -0.8769350000       -1.2560150000
 C        2.6728020000       -0.2844750000        1.0557580000
 C        4.0857830000       -0.4771610000        1.2441690000
 C        4.2736100000       -1.0779170000       -1.0648300000
 C        4.8808480000       -0.8766340000        0.1913270000
 H        2.4404960000       -1.0555350000       -2.2566640000
 H        4.8774540000       -1.3915490000       -1.9250380000
 H        5.9565430000       -1.0338300000        0.3168600000
 H        2.0914700000        0.0274170000        1.9313430000
 H        4.5124910000       -0.3039210000        2.2377540000


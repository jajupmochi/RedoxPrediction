%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 3y_2c_ox

 3 2
 C        0.0000000000       -5.7332320000       -0.6633290000
 C        0.0000000000       -4.9443240000        0.4553040000
 N        0.0000000000       -3.5639870000        0.3520460000
 C        0.0000000000       -2.9421900000       -0.8988830000
 C        0.0000000000       -5.1348960000       -1.9391460000
 C        0.0000000000       -3.7508140000       -2.0445610000
 C        0.0000000000       -2.8106300000        1.4846740000
 C        0.0000000000       -1.4132160000        1.4464480000
 C        0.0000000000       -0.7327290000        0.1744740000
 C        0.0000000000       -1.5236630000       -0.9552470000
 C        0.0000000000       -0.6941340000        2.6701310000
 C        0.0000000000        0.6941340000        2.6701310000
 C        0.0000000000        1.4132160000        1.4464480000
 C        0.0000000000        0.7327290000        0.1744740000
 C        0.0000000000        2.8106300000        1.4846740000
 C        0.0000000000        1.5236630000       -0.9552470000
 C        0.0000000000        2.9421900000       -0.8988830000
 N        0.0000000000        3.5639870000        0.3520460000
 C        0.0000000000        4.9443240000        0.4553040000
 C        0.0000000000        5.7332320000       -0.6633290000
 C        0.0000000000        3.7508140000       -2.0445610000
 C        0.0000000000        5.1348960000       -1.9391460000
 H        0.0000000000       -6.8203650000       -0.5454330000
 H        0.0000000000       -5.3485690000        1.4702860000
 H        0.0000000000       -5.7566420000       -2.8394270000
 H        0.0000000000       -3.2678780000       -3.0244330000
 H        0.0000000000       -1.2331740000        3.6223390000
 H        0.0000000000        1.2331740000        3.6223390000
 H        0.0000000000       -3.3584640000        2.4309590000
 H        0.0000000000       -1.0927610000       -1.9578040000
 H        0.0000000000        1.0927610000       -1.9578040000
 H        0.0000000000        3.3584640000        2.4309590000
 H        0.0000000000        5.3485690000        1.4702860000
 H        0.0000000000        6.8203650000       -0.5454330000
 H        0.0000000000        3.2678780000       -3.0244330000
 H        0.0000000000        5.7566420000       -2.8394270000


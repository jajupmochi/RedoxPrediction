%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 7k_n_ox

 1 2
 C        0.0000000000        3.6305480000       -1.8227230000
 C        0.0000000000        3.8167590000       -0.4105000000
 C        0.0000000000        2.3827800000       -2.3777210000
 C        0.0000000000        1.2308480000       -1.5471620000
 C        0.0000000000        1.3724160000       -0.1504080000
 C        0.0000000000        2.6744830000        0.4473970000
 H        0.0000000000        4.5110540000       -2.4693450000
 H        0.0000000000        2.2555260000       -3.4630990000
 N       -0.0000000000        0.0000000000       -2.1436810000
 C       -0.0000000000       -1.2308480000       -1.5471620000
 C       -0.0000000000       -1.3724160000       -0.1504080000
 S        0.0000000000       -0.0000000000        0.9141780000
 C       -0.0000000000       -2.3827800000       -2.3777210000
 C       -0.0000000000       -3.6305480000       -1.8227230000
 C       -0.0000000000       -3.8167590000       -0.4105000000
 C       -0.0000000000       -2.6744830000        0.4473970000
 H       -0.0000000000       -2.2555260000       -3.4630990000
 H       -0.0000000000       -4.5110540000       -2.4693450000
 H       -0.0000000000        0.0000000000       -3.1592730000
 C        0.0000000000        5.1090790000        0.1571530000
 C        0.0000000000        2.8773670000        1.8479410000
 C        0.0000000000        5.2773420000        1.5266920000
 C        0.0000000000        4.1527420000        2.3737780000
 C       -0.0000000000       -2.8773670000        1.8479410000
 C       -0.0000000000       -5.1090790000        0.1571530000
 C       -0.0000000000       -5.2773420000        1.5266920000
 C       -0.0000000000       -4.1527420000        2.3737780000
 H        0.0000000000       -5.9765550000       -0.5070350000
 H        0.0000000000       -6.2814870000        1.9557830000
 H        0.0000000000       -2.0229250000        2.5288790000
 H        0.0000000000       -4.2896330000        3.4572250000
 H        0.0000000000        4.2896330000        3.4572250000
 H        0.0000000000        2.0229250000        2.5288790000
 H        0.0000000000        5.9765550000       -0.5070350000
 H        0.0000000000        6.2814870000        1.9557830000


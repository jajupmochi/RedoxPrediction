%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 1-chloro-3h_2c_ox

 3 2
 C        2.8343190000        1.3583670000       -0.2541450000
 C        2.6865670000       -0.0047520000        0.0309120000
 C        1.7012060000        2.0957480000       -0.4873330000
 C        1.3545130000       -0.5759990000        0.0314330000
 N        0.4358210000        1.5850330000       -0.4065390000
 C        0.2001660000        0.2812150000       -0.1106780000
 C        1.2142040000       -1.9691790000        0.0748820000
 C       -0.0454730000       -2.5562000000       -0.0840730000
 C       -1.2153090000       -1.7662830000       -0.1379880000
 C       -1.1530920000       -0.3353640000       -0.0148550000
 C       -2.4538940000       -2.4463610000       -0.2554740000
 C       -3.6324260000       -1.7261960000       -0.2174030000
 C       -3.5314920000       -0.3705940000        0.0350690000
 N       -2.3536710000        0.3048370000        0.1633220000
 H        3.8201250000        1.8325870000       -0.3009230000
 H        1.7700940000        3.1568700000       -0.7421870000
 H        2.0994810000       -2.6061680000        0.1676630000
 H       -0.1306380000       -3.6471230000       -0.1402010000
 C       -2.6196320000        1.7081150000        0.6522840000
 C       -1.4682560000        2.6676410000        0.6652310000
 H       -3.0471310000        1.5906230000        1.6626690000
 H       -3.4165710000        2.0852950000       -0.0052390000
 C       -0.6518070000        2.5656120000       -0.6094550000
 H       -0.8221210000        2.5515110000        1.5498560000
 H       -1.9017730000        3.6774150000        0.7496080000
 H       -0.1809600000        3.5191580000       -0.8734650000
 H       -1.2468200000        2.2499000000       -1.4835930000
 H       -4.6170830000       -2.1920300000       -0.3242380000
 H       -4.4321810000        0.2347160000        0.1784950000
 H       -2.4630230000       -3.5360850000       -0.3693710000
Cl        4.0397640000       -0.9235510000        0.3366480000


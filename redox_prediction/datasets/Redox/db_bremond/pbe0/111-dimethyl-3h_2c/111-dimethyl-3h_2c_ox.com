%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 111-dimethyl-3h_2c_ox

 3 2
 C        3.5561220000        0.0046020000       -0.2614570000
 C        2.8775350000       -1.1526090000        0.1408710000
 C        2.8439750000        1.1386390000       -0.5696270000
 C        1.4337910000       -1.0782230000        0.1496090000
 N        1.4884000000        1.2255240000       -0.4600010000
 C        0.7385890000        0.1609650000       -0.0745550000
 C        0.6990400000       -2.2690640000        0.2652410000
 C       -0.6884400000       -2.2795030000        0.0658110000
 C       -1.4252400000       -1.0913430000       -0.0633280000
 C       -0.7472290000        0.1813950000        0.0375440000
 C       -2.8602510000       -1.1886020000       -0.2000040000
 C       -3.5706160000        0.0156210000       -0.1186850000
 C       -2.8938850000        1.1813570000        0.1261750000
 N       -1.5300080000        1.2828100000        0.2198360000
 H        4.6474090000        0.0283840000       -0.3362210000
 H        3.3505390000        2.0474790000       -0.9054740000
 H        1.2137510000       -3.2181290000        0.4346910000
 H       -1.2071060000       -3.2413740000        0.0565110000
 C       -1.1499690000        2.6657830000        0.6800000000
 C        0.2916180000        3.0714590000        0.5709960000
 H       -1.5088210000        2.7392520000        1.7213220000
 H       -1.7715940000        3.3368680000        0.0701060000
 C        0.9096460000        2.5595660000       -0.7160330000
 H        0.8921100000        2.7526260000        1.4372710000
 H        0.3127310000        4.1728510000        0.5874700000
 H        1.7239910000        3.2037200000       -1.0663270000
 H        0.1865490000        2.4784960000       -1.5450600000
 H       -4.6601570000        0.0451780000       -0.2124710000
 H       -3.4398230000        2.1177500000        0.2724010000
 C       -3.5851280000       -2.4612960000       -0.3823090000
 H       -3.0378520000       -3.2181980000       -0.9601990000
 H       -4.5622830000       -2.2878950000       -0.8567650000
 H       -3.8112380000       -2.8987310000        0.6141510000
 C        3.6346130000       -2.3750640000        0.4767430000
 H        4.6511060000       -2.1216680000        0.8131040000
 H        3.7703320000       -2.9874310000       -0.4403740000
 H        3.1565860000       -3.0096100000        1.2350610000

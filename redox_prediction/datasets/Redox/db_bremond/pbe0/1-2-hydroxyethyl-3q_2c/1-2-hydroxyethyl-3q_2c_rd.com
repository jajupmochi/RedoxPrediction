%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 1-2-hydroxyethyl-3q_2c_rd

 1 2
 N        2.7478110000        0.3955080000       -0.4525670000
 C        2.1312720000       -0.8128510000       -0.5979740000
 C        0.7765350000       -0.9436730000       -0.4807230000
 C        1.9876800000        1.4916410000       -0.1712500000
 C        0.6304880000        1.4154160000       -0.0424740000
 C       -0.0633330000        0.1756500000       -0.1938910000
 C       -1.4830770000        0.0661040000       -0.0719650000
 C       -2.1786920000       -1.1733910000       -0.2255420000
 C       -3.5361690000       -1.2505040000       -0.1057090000
 N       -4.2996660000       -0.1535420000        0.1706520000
 C       -2.3247730000        1.1855510000        0.2143060000
 C       -3.6788580000        1.0520920000        0.3235820000
 H        2.7880550000       -1.6595760000       -0.7944620000
 H        0.3645150000       -1.9426310000       -0.6171030000
 H        2.5249670000        2.4350920000       -0.0628780000
 H        0.1005060000        2.3419820000        0.1743890000
 H       -1.6479390000       -2.0998840000       -0.4407690000
 H       -4.0738400000       -2.1926090000       -0.2224980000
 H       -4.3274760000        1.9019910000        0.5408740000
 H       -1.9137390000        2.1839810000        0.3581760000
 C       -5.7496350000       -0.2539910000        0.2233900000
 H       -6.0381490000       -1.2434240000        0.5998640000
 H       -6.1898640000       -0.1069620000       -0.7753650000
 H       -6.1478990000        0.5072310000        0.9057890000
 C        4.2020890000        0.4942890000       -0.5478110000
 C        4.9140570000       -0.0083990000        0.6932650000
 H        4.4630090000        1.5443380000       -0.7360460000
 H        4.5297870000       -0.0931280000       -1.4188220000
 O        4.6328230000       -1.3700240000        0.8245550000
 H        4.5789470000        0.5776830000        1.5718280000
 H        5.9945100000        0.1917000000        0.5562950000
 H        5.0295190000       -1.7069560000        1.6344580000

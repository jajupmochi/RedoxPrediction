%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 49-dimethyl-3c_2c_rd

 1 2
 C       -3.5082380000        0.6599800000       -0.1325440000
 C       -2.8299220000        1.8918900000       -0.2748410000
 C       -1.4609590000        1.9239900000       -0.2217070000
 C       -2.8001290000       -0.5015380000        0.0591600000
 N       -1.4251280000       -0.4526210000        0.1054230000
 C       -0.7142710000        0.7367020000       -0.0304130000
 C        0.7142710000        0.7367020000        0.0304090000
 N        1.4251280000       -0.4526220000       -0.1054170000
 C       -0.6371680000       -1.6356800000        0.4068740000
 C        0.6371680000       -1.6356860000       -0.4068500000
 H       -1.2115960000       -2.5363090000        0.1692080000
 H       -0.4007420000       -1.6579300000        1.4854250000
 H        0.4007420000       -1.6579540000       -1.4854000000
 H        1.2115970000       -2.5363120000       -0.1691690000
 C        1.4609590000        1.9239910000        0.2216940000
 C        2.8299230000        1.8918920000        0.2748250000
 C        3.5082380000        0.6599810000        0.1325330000
 C        2.8001290000       -0.5015380000       -0.0591600000
 H       -4.5962980000        0.6129100000       -0.1740640000
 H        4.5962990000        0.6129120000        0.1740490000
 H       -0.9343960000        2.8657360000       -0.3656830000
 H        0.9343950000        2.8657380000        0.3656630000
 H        3.3938400000        2.8118200000        0.4392700000
 H       -3.3938390000        2.8118170000       -0.4392940000
 C        3.4876710000       -1.8164260000       -0.2308860000
 H        3.2517460000       -2.2838060000       -1.1999650000
 H        3.2214340000       -2.5306320000        0.5652010000
 H        4.5727770000       -1.6669410000       -0.1887630000
 C       -3.4876710000       -1.8164240000        0.2308930000
 H       -4.5727770000       -1.6669410000        0.1887540000
 H       -3.2517580000       -2.2837910000        1.1999820000
 H       -3.2214230000       -2.5306410000       -0.5651800000


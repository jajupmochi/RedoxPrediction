%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 11-dimethyl-22-bipyr_2c_ox

 3 2
 C       -2.7749140000       -1.5069620000        0.6869280000
 C       -3.5096480000       -0.4186810000        0.2293330000
 C       -1.3882240000       -1.4775090000        0.5246080000
 C       -2.8311820000        0.6515290000       -0.4082210000
 N       -1.4949190000        0.6992260000       -0.5146960000
 C       -0.7335320000       -0.3318490000       -0.0106190000
 C        0.7335250000       -0.3318580000        0.0106410000
 C        1.3882000000       -1.4775240000       -0.5245920000
 C        2.7748860000       -1.5069810000       -0.6869450000
 C        3.5096340000       -0.4186960000       -0.2293830000
 N        1.4949270000        0.6992140000        0.5146980000
 C        2.8311870000        0.6515190000        0.4081830000
 H       -3.2647540000       -2.3725680000        1.1485000000
 H       -0.7862290000       -2.3208210000        0.8791850000
 H       -4.6040760000       -0.3806580000        0.2999190000
 H       -3.3904780000        1.4735870000       -0.8731610000
 H        0.7861950000       -2.3208360000       -0.8791510000
 H        3.2647130000       -2.3725930000       -1.1485200000
 H        3.3904960000        1.4735780000        0.8731050000
 H        4.6040600000       -0.3806740000       -0.3000040000
 C       -0.9223980000        1.8216230000       -1.3045030000
 H       -1.4844290000        1.9021550000       -2.2463530000
 H        0.1235340000        1.6146500000       -1.5515820000
 H       -1.0203680000        2.7605830000       -0.7416300000
 C        0.9224420000        1.8215940000        1.3045570000
 H       -0.1235460000        1.6147260000        1.5514830000
 H        1.0205900000        2.7605990000        0.7417910000
 H        1.4843680000        1.9019540000        2.2464850000

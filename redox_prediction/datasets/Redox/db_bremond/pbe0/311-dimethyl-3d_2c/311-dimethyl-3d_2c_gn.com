%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 311-dimethyl-3d_2c_gn

 2 1
 C        3.5492740000       -0.4754370000        0.0518650000
 C        2.8356730000       -1.5697040000        0.5599840000
 C        2.7890770000        0.5758490000       -0.4616380000
 N        1.4425230000        0.5588410000       -0.4757860000
 C        1.4429190000       -1.5769770000        0.5471920000
 C        0.7417090000       -0.4952860000        0.0268820000
 C       -0.7417090000       -0.4952860000       -0.0268820000
 N       -1.4425220000        0.5588410000        0.4757860000
 C       -1.4429200000       -1.5769760000       -0.5471920000
 C       -2.8356730000       -1.5697030000       -0.5599850000
 C       -3.5492740000       -0.4754360000       -0.0518660000
 C       -2.7890770000        0.5758500000        0.4616380000
 C        0.7028680000        1.6968330000       -1.0608860000
 C        0.0000000000        2.5357100000       -0.0000000000
 H       -0.0125230000        1.2741090000       -1.7823840000
 H        1.4121760000        2.3034240000       -1.6372310000
 C       -0.7028680000        1.6968330000        1.0608850000
 H       -0.7126400000        3.1955400000       -0.5188750000
 H        0.7126400000        3.1955400000        0.5188740000
 H       -1.4121760000        2.3034240000        1.6372310000
 H        0.0125230000        1.2741090000        1.7823840000
 H       -0.8909850000       -2.4229600000       -0.9617750000
 H       -3.3744260000       -2.4259400000       -0.9762810000
 H       -3.2599380000        1.4632740000        0.8908660000
 H        0.8909840000       -2.4229600000        0.9617740000
 H        3.3744250000       -2.4259410000        0.9762790000
 H        3.2599380000        1.4632730000       -0.8908670000
 C        5.0402520000       -0.4258430000        0.0457370000
 H        5.4332420000       -0.5360020000        1.0686550000
 H        5.4241340000        0.5130410000       -0.3729670000
 H        5.4483740000       -1.2601200000       -0.5467720000
 C       -5.0402520000       -0.4258440000       -0.0457350000
 H       -5.4332470000       -0.5360480000       -1.0686470000
 H       -5.4241340000        0.5130570000        0.3729320000
 H       -5.4483700000       -1.2600960000        0.5468120000

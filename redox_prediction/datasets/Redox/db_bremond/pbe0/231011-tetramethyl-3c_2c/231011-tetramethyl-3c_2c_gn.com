%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 231011-tetramethyl-3c_2c_gn

 2 1
 C        3.5398480000       -0.0678080000       -0.0858660000
 C        2.8922220000        1.1672700000        0.1710190000
 C        1.4914210000        1.1834120000        0.2092130000
 C        2.7393260000       -1.1866700000       -0.2679350000
 N        1.3925650000       -1.1348240000       -0.2232390000
 C        0.7380770000        0.0355480000       -0.0009070000
 C       -0.7380770000        0.0355480000        0.0009070000
 N       -1.3925650000       -1.1348230000        0.2232410000
 C        0.6000080000       -2.3504250000       -0.4568560000
 C       -0.6000080000       -2.3504250000        0.4568590000
 H        1.2280050000       -3.2267860000       -0.2525540000
 H        0.3015510000       -2.3802590000       -1.5179130000
 H       -0.3015510000       -2.3802570000        1.5179160000
 H       -1.2280050000       -3.2267850000        0.2525580000
 C       -1.4914210000        1.1834120000       -0.2092150000
 C       -2.8922220000        1.1672700000       -0.1710200000
 C       -3.5398480000       -0.0678080000        0.0858660000
 C       -2.7393260000       -1.1866690000        0.2679370000
 H        3.1721880000       -2.1717040000       -0.4573430000
 H       -3.1721870000       -2.1717030000        0.4573460000
 C       -3.6772250000        2.4081940000       -0.4008300000
 H       -3.0379780000        3.2783660000       -0.5926760000
 H       -4.3160850000        2.6276400000        0.4708830000
 H       -4.3613390000        2.2807730000       -1.2562980000
 C       -5.0249930000       -0.1856070000        0.1472730000
 H       -5.4822570000        0.1344840000       -0.8028360000
 H       -5.4353500000        0.4692190000        0.9327730000
 H       -5.3506190000       -1.2127450000        0.3545340000
 C        3.6772250000        2.4081940000        0.4008270000
 H        3.0379780000        3.2783690000        0.5926640000
 H        4.3160910000        2.6276350000       -0.4708820000
 H        4.3613320000        2.2807760000        1.2563010000
 C        5.0249930000       -0.1856070000       -0.1472730000
 H        5.3506190000       -1.2127470000       -0.3545240000
 H        5.4822580000        0.1344930000        0.8028320000
 H        5.4353480000        0.4692120000       -0.9327790000
 H        0.9813950000        2.1212890000        0.4314220000
 H       -0.9813950000        2.1212890000       -0.4314250000

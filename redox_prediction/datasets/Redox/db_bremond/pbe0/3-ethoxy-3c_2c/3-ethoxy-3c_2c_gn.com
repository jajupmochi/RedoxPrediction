%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 3-ethoxy-3c_2c_gn

 2 1
 C       -4.7094590000       -0.1713720000        0.0633490000
 C       -4.1171540000       -1.4134020000        0.2894890000
 C       -2.7285480000       -1.5274120000        0.2713740000
 C       -3.8921840000        0.9204960000       -0.1631300000
 N       -2.5493660000        0.7925300000       -0.1797860000
 C       -1.9362540000       -0.4085270000        0.0204020000
 C       -0.4673950000       -0.4776540000       -0.0445240000
 N        0.2619280000        0.6524730000        0.1375290000
 C       -1.7023330000        1.9614450000       -0.4614540000
 C       -0.4640550000        1.9080300000        0.3972470000
 H       -2.2719630000        2.8734090000       -0.2414880000
 H       -1.4521670000        1.9616760000       -1.5355060000
 H       -0.7083100000        1.9532350000        1.4715450000
 H        0.1929500000        2.7535390000        0.1581570000
 C        0.2234710000       -1.6750660000       -0.2725050000
 C        1.6023820000       -1.7051370000       -0.2862150000
 C        2.3416090000       -0.5207530000       -0.0679740000
 C        1.6074640000        0.6643240000        0.1344730000
 H       -5.7934830000       -0.0390350000        0.0720930000
 H       -2.2604650000       -2.4905480000        0.4761870000
 H       -0.3313940000       -2.5937730000       -0.4652370000
 H       -4.7337570000       -2.2938490000        0.4892580000
 O        3.6371550000       -0.5977220000       -0.0824590000
 C        4.4865470000        0.5579480000        0.1294590000
 C        5.9180330000        0.1076480000        0.0631040000
 H        4.2372640000        0.9843850000        1.1164780000
 H        4.2537980000        1.2969380000       -0.6566410000
 H        6.1362110000       -0.6346800000        0.8431790000
 H        6.5753250000        0.9744840000        0.2216180000
 H        6.1521240000       -0.3245060000       -0.9195580000
 H        2.1492010000       -2.6334740000       -0.4709760000
 H        2.0843040000        1.6314010000        0.2945250000
 H       -4.2875740000        1.9241390000       -0.3367360000

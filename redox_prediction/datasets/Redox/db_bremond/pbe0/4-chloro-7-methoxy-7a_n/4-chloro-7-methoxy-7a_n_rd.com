%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 4-chloro-7-methoxy-7a_n_rd

-1 2
 C       -3.4370370000        1.0721350000       -0.1499530000
 C       -3.4258990000       -0.3303850000       -0.0490710000
 C       -2.2247880000        1.7644940000       -0.0906280000
 C       -0.9940620000        1.1204200000        0.0833050000
 C       -1.0017070000       -0.3049400000        0.2336820000
 C       -2.2135890000       -0.9935030000        0.1356150000
 H       -4.3637090000        1.6319990000       -0.2814810000
 H       -2.2349710000        2.8569740000       -0.1825330000
 H       -2.2216940000       -2.0831620000        0.2243280000
 N        0.1787430000        1.8445420000        0.1355760000
 C        1.4789530000        1.3611190000        0.0660980000
 C        1.6908540000       -0.0664310000       -0.0053940000
 S        0.4569770000       -1.1462700000        0.7157360000
 C        2.5611750000        2.2336240000        0.0362840000
 C        3.8775930000        1.7691650000       -0.1164120000
 C        4.1025170000        0.3762350000       -0.2499920000
 C        3.0252730000       -0.4877480000       -0.1770170000
 H        2.3693620000        3.3125410000        0.1026570000
 H        4.7144700000        2.4710640000       -0.1315680000
 H        5.0989880000       -0.0255040000       -0.4476370000
 H        0.0793580000        2.8485850000        0.0890670000
Cl        3.3361780000       -2.2205710000       -0.3330320000
 O       -4.5502430000       -1.1196320000       -0.1300580000
 C       -5.7658170000       -0.4938190000       -0.3535010000
 H       -6.5332390000       -1.2809760000       -0.4047280000
 H       -5.7840960000        0.0743440000       -1.3050250000
 H       -6.0411860000        0.2072140000        0.4600290000


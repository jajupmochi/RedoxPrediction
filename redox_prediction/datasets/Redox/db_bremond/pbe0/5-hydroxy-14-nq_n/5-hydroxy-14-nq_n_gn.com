%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 5-hydroxy-14-nq_n_gn

 0 1
 C        1.7198180000       -1.2482330000        0.0000000000
 C        2.6987710000       -0.2441550000        0.0000000000
 C        0.3800390000       -0.8905630000        0.0000000000
 C        0.0000000000        0.4709830000        0.0000000000
 C        0.9942760000        1.4784850000        0.0000000000
 C        2.3504170000        1.0955980000        0.0000000000
 H        3.1050480000        1.8845000000        0.0000000000
 H        3.7560240000       -0.5210520000        0.0000000000
 H        1.9802140000       -2.3080320000        0.0000000000
 C       -0.6620460000       -1.9564230000        0.0000000000
 C       -2.0780840000       -1.5131420000        0.0000000000
 C       -2.4228120000       -0.2152960000        0.0000000000
 C       -1.4089540000        0.8580340000        0.0000000000
 O       -0.3844780000       -3.1387270000        0.0000000000
 O       -1.7631360000        2.0401640000        0.0000000000
 O        0.6875750000        2.7625690000        0.0000000000
 H       -0.3044660000        2.8179630000        0.0000000000
 H       -2.8211490000       -2.3148970000        0.0000000000
 H       -3.4639180000        0.1177390000        0.0000000000

%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 3-chloro-7a_n_ox

 1 2
 C        1.6862750000       -4.1720280000        0.0000000000
 C        2.8857750000       -3.4378190000        0.0000000000
 C        0.4687940000       -3.5216710000        0.0000000000
 C        0.4176350000       -2.1141750000        0.0000000000
 C        1.6275340000       -1.3765130000        0.0000000000
 C        2.8563030000       -2.0541940000        0.0000000000
 H        1.7147460000       -5.2634850000        0.0000000000
 H       -0.4641760000       -4.0914260000        0.0000000000
 H        3.8455470000       -3.9580860000        0.0000000000
 H        3.7870310000       -1.4815960000        0.0000000000
 N       -0.7998270000       -1.4857020000        0.0000000000
 C       -1.0512550000       -0.1431160000        0.0000000000
 C        0.0000000000        0.8098570000        0.0000000000
 S        1.6757170000        0.3584790000        0.0000000000
 C       -2.3857010000        0.3133150000        0.0000000000
 C       -2.6682210000        1.6606730000        0.0000000000
 C       -1.6159570000        2.6009190000        0.0000000000
 C       -0.2933630000        2.1784340000        0.0000000000
 H       -3.2017450000       -0.4139980000        0.0000000000
 H       -3.7016140000        2.0112770000        0.0000000000
 H        0.5119720000        2.9159280000        0.0000000000
 H       -1.6147720000       -2.0936270000        0.0000000000
Cl       -1.9797990000        4.2692450000        0.0000000000

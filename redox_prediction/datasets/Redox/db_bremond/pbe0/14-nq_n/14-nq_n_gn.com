%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 14-nq_n_gn

 0 1
 C        2.6714630000        0.6987950000       -0.0000130000
 C        1.4682080000        1.3983510000       -0.0000030000
 C        0.2574620000        0.7028340000        0.0000070000
 C        0.2574620000       -0.7028340000        0.0000100000
 C        1.4682080000       -1.3983510000       -0.0000000000
 C        2.6714630000       -0.6987950000       -0.0000120000
 H        3.6184660000       -1.2439380000       -0.0000180000
 H        3.6184660000        1.2439380000       -0.0000220000
 H        1.4367570000        2.4901340000       -0.0000050000
 H        1.4367570000       -2.4901340000        0.0000030000
 C       -1.0233210000        1.4640080000        0.0000110000
 C       -2.2762840000        0.6714250000       -0.0000020000
 C       -1.0233210000       -1.4640080000        0.0000310000
 C       -2.2762840000       -0.6714250000       -0.0000010000
 O       -1.0550220000        2.6781550000        0.0000090000
 O       -1.0550230000       -2.6781550000       -0.0000200000
 H       -3.2002140000       -1.2559710000       -0.0000150000
 H       -3.2002140000        1.2559710000       -0.0000140000


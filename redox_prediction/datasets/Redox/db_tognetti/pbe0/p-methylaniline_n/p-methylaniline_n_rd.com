%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 p-methylaniline_n_rd

-1 2
 C        0.2179700000       -0.0447600000       -0.2830000000
 N        0.2419000000       -0.0658400000       -1.6888000000
 C        0.1853800000        1.1677900000        0.4196500000
 H        0.2353200000        2.1077800000       -0.1263900000
 C        0.0858500000        1.1781400000        1.8098400000
 H        0.0650000000        2.1348100000        2.3281300000
 C        0.0140400000       -0.0052700000        2.5521700000
 C       -0.0949800000        0.0126600000        4.0594900000
 C        0.0482400000       -1.2120300000        1.8396500000
 H       -0.0023300000       -2.1554100000        2.3802300000
 C        0.1471000000       -1.2390900000        0.4523200000
 H        0.1669400000       -2.1933100000       -0.0706300000
 H       -0.1021600000        1.0391600000        4.4387200000
 H        0.7457000000       -0.5125600000        4.5293400000
 H       -1.0153900000       -0.4769400000        4.4007900000
 H        0.6662900000        0.7566600000       -2.1044300000
 H        0.6439900000       -0.9112900000       -2.0797700000


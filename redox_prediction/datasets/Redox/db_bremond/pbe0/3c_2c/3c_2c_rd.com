%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 3c_2c_rd

 1 2
 C       -3.5227130000       -0.3058080000        0.0371510000
 C       -2.8647840000       -1.5575170000        0.1549350000
 C       -1.4941620000       -1.6101380000        0.1367150000
 C       -2.7619500000        0.8178380000       -0.1078870000
 N       -1.4005960000        0.7645240000       -0.1363460000
 C       -0.7133830000       -0.4342430000       -0.0001390000
 C        0.7133830000       -0.4342430000        0.0001390000
 N        1.4005960000        0.7645240000        0.1363460000
 C       -0.6307830000        1.9655400000       -0.4171410000
 C        0.6307830000        1.9655400000        0.4171410000
 H       -1.2420360000        2.8459460000       -0.1799880000
 H       -0.3802880000        2.0005480000       -1.4915430000
 H        0.3802880000        2.0005480000        1.4915430000
 H        1.2420360000        2.8459460000        0.1799880000
 C        1.4941620000       -1.6101380000       -0.1367150000
 C        2.8647840000       -1.5575170000       -0.1549340000
 C        3.5227130000       -0.3058080000       -0.0371510000
 C        2.7619500000        0.8178380000        0.1078870000
 H       -4.6093220000       -0.2236290000        0.0624940000
 H       -3.1970600000        1.8132070000       -0.2152170000
 H        4.6093220000       -0.2236290000       -0.0624940000
 H        3.1970600000        1.8132070000        0.2152170000
 H       -3.4446900000       -2.4744780000        0.2742450000
 H       -0.9894810000       -2.5672960000        0.2608490000
 H        0.9894810000       -2.5672960000       -0.2608480000
 H        3.4446900000       -2.4744780000       -0.2742440000


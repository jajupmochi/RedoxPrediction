%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 14-nq-6-ch2cl_n_rd

-1 2
 C        1.2289230000       -1.4001790000        0.4143250000
 C        1.6007890000       -0.0348500000        0.4634000000
 C       -0.0909070000       -1.7546450000        0.2207890000
 C       -1.0959170000       -0.7779720000        0.0713980000
 C       -0.7309440000        0.5919860000        0.1159570000
 C        0.6191340000        0.9333000000        0.3055200000
 C       -2.4932520000       -1.2007660000       -0.1234970000
 C       -3.4360530000       -0.1194320000       -0.2630130000
 C       -3.0824600000        1.2117030000       -0.2187210000
 C       -1.7292170000        1.6682700000       -0.0272560000
 H        1.9994100000       -2.1704640000        0.5221430000
 H       -0.4126950000       -2.7986780000        0.1791200000
 H        0.8529000000        2.0023200000        0.3320910000
 H       -4.4798550000       -0.4167110000       -0.4097280000
 H       -3.8381120000        1.9966310000       -0.3287360000
 C        3.0163960000        0.3589560000        0.7083870000
Cl        4.1351560000       -0.0283080000       -0.6811210000
 H        3.4544830000       -0.1727140000        1.5650130000
 H        3.1073450000        1.4402350000        0.8716760000
 O       -2.8191890000       -2.4070390000       -0.1618310000
 O       -1.4083200000        2.8748400000        0.0172990000

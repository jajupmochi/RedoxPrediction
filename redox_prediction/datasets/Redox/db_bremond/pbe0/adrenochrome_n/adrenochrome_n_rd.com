%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 adrenochrome_n_rd

-1 2
 C       -2.0458010000        0.8121250000        0.1270570000
 C       -2.1582570000       -0.6861720000       -0.1373230000
 C       -0.7153140000        1.3816030000        0.1909680000
 C        0.4012600000        0.5883280000        0.0350450000
 C        0.3041450000       -0.8062340000       -0.2371300000
 C       -0.9309080000       -1.4256290000       -0.3015170000
 H       -0.6546030000        2.4578910000        0.3752550000
 H       -1.0267800000       -2.4991990000       -0.4964820000
 O       -3.2766930000       -1.2286270000       -0.2073590000
 O       -3.0672340000        1.4960020000        0.2706000000
 N        1.7526820000        0.9489820000        0.1169260000
 C        2.5395120000       -0.1147210000       -0.4721000000
 C        2.1687630000        2.2990560000       -0.0776120000
 H        1.5815270000        2.9738840000        0.5635920000
 H        2.0525130000        2.6525010000       -1.1280960000
 H        3.2297320000        2.4107750000        0.2011290000
 C        1.6895520000       -1.3826420000       -0.2838630000
 H        3.5218300000       -0.2210640000        0.0192010000
 H        2.7111760000        0.0658650000       -1.5577180000
 O        2.0345470000       -2.0952170000        0.8986460000
 H        1.8517760000       -2.0845350000       -1.1201290000
 H        1.4213840000       -1.7705460000        1.5685190000

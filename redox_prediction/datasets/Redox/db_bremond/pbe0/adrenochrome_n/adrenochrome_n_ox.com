%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 adrenochrome_n_ox

 1 2
 C       -2.0870200000        0.8515080000       -0.1730940000
 C       -2.0556480000       -0.8674560000        0.1577010000
 C       -0.8329320000        1.4736520000       -0.0197730000
 C        0.3288860000        0.7135650000       -0.0497360000
 C        0.3305050000       -0.7433090000       -0.2181680000
 C       -0.7721810000       -1.4991710000       -0.1266560000
 H       -0.8423320000        2.5527780000        0.1419640000
 H       -0.7597120000       -2.5910220000       -0.1925160000
 O       -3.0589890000       -1.3357960000        0.5468320000
 O       -3.1892360000        1.2536770000       -0.3692030000
 N        1.5885230000        1.1221170000        0.0593270000
 C        2.5447930000        0.0195220000        0.1071080000
 C        2.0042640000        2.4923000000        0.2557270000
 H        1.3229920000        3.1763660000       -0.2665160000
 H        3.0146400000        2.6250910000       -0.1525760000
 H        2.0223850000        2.7460430000        1.3288200000
 C        1.7487610000       -1.1886380000       -0.4318510000
 H        2.8496500000       -0.1834070000        1.1484320000
 H        3.4388260000        0.2676030000       -0.4825190000
 O        1.9947630000       -2.4005960000        0.1786600000
 H        1.8917040000       -1.2339920000       -1.5324130000
 H        2.7133150000       -2.8643960000       -0.2658180000

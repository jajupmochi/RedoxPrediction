%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 3e_2c_gn

 2 1
 C       -3.5069870000       -0.7985520000        0.1776890000
 C       -2.7696580000       -1.7864840000        0.8224690000
 C       -1.3753350000       -1.7424310000        0.7534890000
 C       -2.8329010000        0.1949280000       -0.5134170000
 N       -1.4856330000        0.2313270000       -0.5717510000
 C       -0.7423830000       -0.7195740000        0.0571700000
 H       -4.5992190000       -0.7912380000        0.1942690000
 H       -3.3612160000        0.9866220000       -1.0497730000
 H       -3.2689900000       -2.5884060000        1.3731400000
 H       -0.7688600000       -2.5032890000        1.2497010000
 C        0.7423830000       -0.7195740000       -0.0571700000
 N        1.4856330000        0.2313270000        0.5717510000
 C       -0.8600940000        1.3546330000       -1.3183690000
 C       -0.6227650000        2.5804760000       -0.4402090000
 H       -1.5532090000        1.5972540000       -2.1340430000
 H        0.0593210000        0.9780540000       -1.7897540000
 C        0.8600940000        1.3546330000        1.3183690000
 C        0.6227650000        2.5804760000        0.4402090000
 H        1.5532090000        1.5972540000        2.1340430000
 H       -0.0593220000        0.9780540000        1.7897540000
 H       -1.5251510000        2.7558900000        0.1697540000
 H       -0.5585720000        3.4478470000       -1.1155290000
 H        1.5251510000        2.7558900000       -0.1697540000
 H        0.5585720000        3.4478470000        1.1155290000
 C        1.3753350000       -1.7424310000       -0.7534900000
 C        2.7696580000       -1.7864840000       -0.8224690000
 C        3.5069870000       -0.7985520000       -0.1776890000
 H        0.7688600000       -2.5032890000       -1.2497010000
 H        3.2689900000       -2.5884060000       -1.3731400000
 C        2.8329010000        0.1949280000        0.5134170000
 H        3.3612160000        0.9866220000        1.0497740000
 H        4.5992190000       -0.7912380000       -0.1942690000

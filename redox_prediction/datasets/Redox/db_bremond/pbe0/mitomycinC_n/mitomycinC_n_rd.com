%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 mitomycinC_n_rd

-1 2
 C        2.9165100000       -1.6670970000        0.3793530000
 C        3.4738480000       -0.6383850000       -0.3581460000
 C        1.5076520000       -1.6802130000        0.7938500000
 C        0.7612490000       -0.5484110000        0.3624990000
 C        1.2908770000        0.4626450000       -0.4137650000
 C        2.6896260000        0.5036110000       -0.8048110000
 O        1.0533370000       -2.6245800000        1.4959300000
 O        3.1865740000        1.4519280000       -1.4692680000
 N        3.6225460000       -2.7926370000        0.7766010000
 C        4.9209240000       -0.6758880000       -0.7443290000
 H        5.5981290000       -0.5825530000        0.1281210000
 H        5.1274210000        0.1675230000       -1.4186430000
 H        5.1838710000       -1.6228080000       -1.2495850000
 C       -0.7029360000       -0.3033540000        0.5838990000
 C       -0.8371290000        1.1773380000        0.1175530000
 N        0.3142880000        1.4127720000       -0.7653480000
 C        0.6971490000        2.8184720000       -0.6454760000
 C        0.3707050000        3.1851310000        0.7970190000
 H        0.0826600000        3.4312710000       -1.3238440000
 H        1.7577400000        2.9100090000       -0.9162770000
 C       -0.6026370000        2.1820700000        1.2750470000
 N       -1.0262910000        3.5598940000        1.0080480000
 H       -0.5969510000        1.8093770000        2.3043020000
 H        1.1400060000        3.5918550000        1.4622860000
 O       -1.9999260000        1.4483640000       -0.6205840000
 C       -3.1709240000        1.6843300000        0.1077230000
 H       -3.1938250000        2.7101130000        0.5124010000
 H       -3.3094090000        0.9617190000        0.9307370000
 H       -4.0146300000        1.5535530000       -0.5873650000
 H       -1.1221480000        4.0480770000        1.8991940000
 H        4.5826070000       -2.6332780000        1.0582200000
 H        3.0617340000       -3.3003440000        1.4621600000
 C       -1.5027920000       -1.3068170000       -0.2291880000
 H       -0.9864350000       -0.4274020000        1.6438840000
 O       -2.8666710000       -1.3238800000        0.2202260000
 H       -1.0710490000       -2.3041750000       -0.0593380000
 H       -1.4755580000       -1.0719340000       -1.3016480000
 C       -3.7709700000       -1.9014100000       -0.5699020000
 O       -3.5502650000       -2.4355780000       -1.6299780000
 N       -5.0214110000       -1.7795450000       -0.0163070000
 H       -5.0858340000       -1.5495610000        0.9653230000
 H       -5.7335470000       -2.3770090000       -0.4094470000

%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 i47di-12-met-3-phenyl-5-met_n_ox

 1 2
 C       -3.3365970000       -1.0715710000        0.1170010000
 C       -2.3689670000       -2.0172250000        0.1264590000
 C       -3.0722790000        0.3741400000        0.0326230000
 C       -1.6401030000        0.7439770000       -0.0551110000
 C       -0.6240600000       -0.1917240000       -0.0846370000
 C       -0.9190800000       -1.6422470000        0.0405950000
 C       -1.0534120000        2.0382700000       -0.0748950000
 C        0.6229600000        0.5271960000       -0.0848770000
 N        0.2993630000        1.8751210000       -0.0820790000
 C       -1.7226420000        3.3495040000       -0.1091200000
 H       -2.7945150000        3.2120970000        0.0777930000
 H       -1.6124020000        3.8109160000       -1.1077990000
 H       -1.2906790000        4.0486220000        0.6233650000
 C        1.2242300000        2.9888190000       -0.2357800000
 H        0.7688760000        3.7474190000       -0.8838720000
 H        2.1477620000        2.6374030000       -0.7084520000
 H        1.4614870000        3.4466900000        0.7353840000
 C        1.9695150000        0.0278760000       -0.0341700000
 C        2.3220070000       -1.1328960000       -0.7691660000
 C        2.9619990000        0.6650840000        0.7563450000
 C        4.2461890000        0.1556060000        0.8090180000
 C        3.6216470000       -1.6111040000       -0.7365210000
 C        4.5835830000       -0.9764480000        0.0536120000
 H        1.5762870000       -1.6317150000       -1.3849530000
 H        3.8893790000       -2.4898330000       -1.3263730000
 H        5.6032050000       -1.3677250000        0.0896910000
 H        2.7027140000        1.5201760000        1.3818640000
 H        4.9940550000        0.6292140000        1.4482210000
 C       -2.6456370000       -3.4765440000        0.2212280000
 H       -3.7218480000       -3.6776050000        0.2849930000
 H       -2.1430350000       -3.9052840000        1.1022910000
 H       -2.2292870000       -4.0048970000       -0.6507460000
 O       -3.9548790000        1.2078870000        0.0362960000
 O       -0.0538590000       -2.4905840000        0.0737220000
 H       -4.3937490000       -1.3440210000        0.1773720000

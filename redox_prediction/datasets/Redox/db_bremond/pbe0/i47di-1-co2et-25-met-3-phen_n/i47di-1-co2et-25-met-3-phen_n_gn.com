%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 i47di-1-co2et-25-met-3-phen_n_gn

 0 1
 C       -1.2355560000        3.0493750000       -0.2530670000
 C        0.0893780000        3.2813390000       -0.1455790000
 C       -1.8642990000        1.7063760000       -0.2967120000
 C       -0.9308150000        0.5802220000       -0.0898950000
 C        0.4703290000        0.8014780000       -0.0083240000
 C        1.0605280000        2.1430110000       -0.0150390000
 H       -1.9455300000        3.8755120000       -0.3563810000
 O       -3.0518200000        1.5992180000       -0.5350180000
 O        2.2521570000        2.3713970000        0.0920230000
 C        1.0972040000       -0.4461860000        0.0253930000
 C       -1.1240910000       -0.7966810000       -0.0786720000
 N        0.1124010000       -1.3979940000       -0.0199470000
 C       -2.3555490000       -1.6116070000       -0.1612770000
 O       -3.3804340000       -1.0115990000        0.4185920000
 O       -2.4068330000       -2.7010570000       -0.6854530000
 C       -4.6689520000       -1.5984050000        0.2395780000
 C       -5.6920630000       -0.6173830000        0.7505280000
 H       -4.8087530000       -1.8250740000       -0.8292910000
 H       -4.7024030000       -2.5598680000        0.7784020000
 H       -5.5521320000       -0.4255620000        1.8245450000
 H       -5.5988420000        0.3386390000        0.2162540000
 H       -6.7060880000       -1.0156370000        0.5994110000
 C        0.3268360000       -2.8296920000        0.0957300000
 H        0.1868590000       -3.3269810000       -0.8729090000
 H        1.3427250000       -3.0054700000        0.4655180000
 H       -0.3978070000       -3.2581500000        0.7980860000
 C        0.6908770000        4.6454930000       -0.1434440000
 H        1.2658710000        4.8092470000        0.7812490000
 H        1.4154300000        4.7456810000       -0.9665700000
 H       -0.0761550000        5.4251330000       -0.2376520000
 C        2.5293600000       -0.7866380000        0.0660130000
 C        3.3573620000       -0.2649800000        1.0693500000
 C        3.0920830000       -1.6252920000       -0.9086460000
 C        4.4470750000       -1.9443290000       -0.8726560000
 C        5.2585190000       -1.4297690000        0.1372110000
 C        4.7099940000       -0.5890830000        1.1053830000
 H        2.9339920000        0.4041000000        1.8194560000
 H        5.3428280000       -0.1767690000        1.8947410000
 H        4.8719750000       -2.5922770000       -1.6429970000
 H        6.3220600000       -1.6789920000        0.1660690000
 H        2.4652900000       -2.0147440000       -1.7147050000

%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 6-hydroxy-3c_2c_rd

 1 2
 C        0.5285440000        1.7984520000        0.4341490000
 C       -2.5050930000       -2.1381140000        0.1501980000
 C       -1.1397720000       -2.0068290000        0.1792910000
 C        2.7747800000        0.9379450000        0.0554360000
 N        1.4304340000        0.7105180000        0.1272610000
 H        3.0809940000        1.9829300000        0.1300390000
 C       -3.3167160000       -0.9908330000       -0.0329140000
 C        0.8982570000       -0.5681430000        0.0272250000
 C       -0.5166150000       -0.7408070000        0.0364600000
 C        3.6664820000       -0.0832070000       -0.0894240000
 C       -2.7086130000        0.2252130000       -0.1650020000
 H       -4.4041460000       -1.0566390000       -0.0695750000
 C        1.8191570000       -1.6384040000       -0.1013690000
 C        3.1708780000       -1.4122510000       -0.1578040000
 N       -1.3536670000        0.3566180000       -0.1171280000
 H        4.7326590000        0.1351290000       -0.1507900000
 C       -0.7560590000        1.6911420000       -0.3703700000
 H       -3.2553670000        1.1562520000       -0.3172100000
 O       -1.6314690000        2.7059020000       -0.0916450000
 H       -0.5294500000        1.7529350000       -1.4488000000
 H       -1.7175540000        2.8350720000        0.8632130000
 H        1.0115000000        2.7550830000        0.1978310000
 H        0.2831720000        1.7900830000        1.5132400000
 H       -2.9610850000       -3.1222320000        0.2731980000
 H       -0.5164500000       -2.8853760000        0.3396970000
 H        3.8603110000       -2.2506310000       -0.2718160000
 H        1.4384390000       -2.6547560000       -0.1920480000


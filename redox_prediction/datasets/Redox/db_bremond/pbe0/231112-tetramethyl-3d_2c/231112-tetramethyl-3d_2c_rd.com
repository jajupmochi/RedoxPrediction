%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 231112-tetramethyl-3d_2c_rd

 1 2
 C        3.5641460000       -0.2667140000       -0.0523740000
 C        2.7304120000       -1.3981180000        0.2393640000
 C        2.9344880000        0.9258980000       -0.2517540000
 N        1.5755340000        1.1225010000       -0.2309400000
 C        1.3718870000       -1.1990030000        0.2573110000
 C        0.7228400000        0.0462500000        0.0117780000
 C       -0.7209810000        0.0225980000        0.0087390000
 N       -1.5493320000        1.0639800000        0.3938360000
 C       -1.3877910000       -1.1789750000       -0.3598770000
 C       -2.7515180000       -1.3530290000       -0.3014580000
 C       -3.5558870000       -0.2583930000        0.1616400000
 C       -2.9025980000        0.9034520000        0.4651550000
 C        1.2294750000        2.5359500000       -0.4631740000
 C       -0.2302510000        2.9056710000       -0.5010130000
 H        1.7131920000        2.8417700000       -1.4036470000
 H        1.7101870000        3.1119550000        0.3457940000
 C       -0.9993530000        2.3737210000        0.6944560000
 H       -0.7194600000        2.5679690000       -1.4282760000
 H       -0.2698800000        4.0051980000       -0.5137500000
 H       -1.8383580000        3.0314060000        0.9501580000
 H       -0.3569390000        2.3077380000        1.5911070000
 H       -0.7845680000       -1.9901670000       -0.7654420000
 H       -3.4517680000        1.7896660000        0.7877040000
 H        0.7399370000       -2.0393700000        0.5406760000
 C       -3.3908770000       -2.6304220000       -0.7384010000
 H       -2.6468000000       -3.3646660000       -1.0731560000
 H       -4.0987880000       -2.4582340000       -1.5652630000
 H       -3.9723710000       -3.0804400000        0.0823290000
 C       -5.0435680000       -0.3595870000        0.2751030000
 H       -5.3361370000       -1.1603120000        0.9728930000
 H       -5.5047060000       -0.5991980000       -0.6962400000
 H       -5.4869650000        0.5781970000        0.6357860000
 C        5.0544150000       -0.3732700000       -0.1080760000
 H        5.4653080000       -0.7319400000        0.8489940000
 H        5.5223330000        0.5941000000       -0.3345000000
 H        5.3728080000       -1.0914880000       -0.8803450000
 C        3.3312100000       -2.7295860000        0.5495920000
 H        3.9487800000       -3.0884990000       -0.2895030000
 H        2.5619720000       -3.4851090000        0.7555980000
 H        3.9973240000       -2.6714780000        1.4257140000
 H        3.5151730000        1.8288880000       -0.4489750000


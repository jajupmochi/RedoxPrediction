%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 11-bis-carbamylmethyl-3l_2c_ox

 3 2
 N       -3.4546480000       -0.0007910000        0.6974500000
 C       -2.8730570000       -1.1048720000        0.1778040000
 C       -1.5172340000       -1.1274600000       -0.1002610000
 C       -2.7244320000        1.1095890000        0.9432940000
 C       -1.3666140000        1.1403480000        0.6786170000
 C       -0.7292150000        0.0084200000        0.1464310000
 C        0.7292150000        0.0084180000       -0.1464200000
 C        1.5172340000       -1.1274580000        0.1002920000
 C        1.3666130000        1.1403340000       -0.6786320000
 C        2.7244300000        1.1095690000       -0.9433120000
 C        2.8730570000       -1.1048760000       -0.1777760000
 N        3.4546460000       -0.0008060000       -0.6974480000
 H        3.2487630000        1.9705790000       -1.3649680000
 H        0.8200090000        2.0542650000       -0.9214000000
 H        1.1014010000       -2.0396880000        0.5338250000
 H        3.5128710000       -1.9715940000        0.0032550000
 H       -0.8200110000        2.0542840000        0.9213670000
 H       -3.2487660000        1.9706090000        1.3649280000
 H       -3.5128710000       -1.9715930000       -0.0032120000
 H       -1.1014000000       -2.0396980000       -0.5337750000
 C       -4.8981550000       -0.0061800000        0.9504750000
 C       -5.7073320000       -0.0176070000       -0.3671440000
 H       -5.1646420000        0.8716050000        1.5557590000
 H       -5.1667840000       -0.9126180000        1.5183550000
 O       -5.1114780000       -0.3915800000       -1.3799300000
 N       -6.9751200000        0.3232030000       -0.3142090000
 C        4.8981520000       -0.0062030000       -0.9504810000
 C        5.7073370000       -0.0176060000        0.3671350000
 H        5.1646380000        0.8715690000       -1.5557820000
 H        5.1667750000       -0.9126530000       -1.5183450000
 N        6.9751230000        0.3232050000        0.3141880000
 O        5.1114810000       -0.3915400000        1.3799350000
 H       -7.4321910000        0.6277840000        0.5416650000
 H       -7.5490870000        0.2620610000       -1.1566590000
 H        7.4321870000        0.6277790000       -0.5416920000
 H        7.5490890000        0.2621010000        1.1566420000


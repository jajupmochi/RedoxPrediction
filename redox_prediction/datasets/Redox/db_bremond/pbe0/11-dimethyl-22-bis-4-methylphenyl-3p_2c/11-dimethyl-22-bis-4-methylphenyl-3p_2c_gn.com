%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 11-dimethyl-22-bis-4-methylphenyl-3p_2c_gn

 2 1
 N        3.4091240000       -1.9343880000       -0.8915780000
 C        2.6472330000       -3.0373160000       -1.0811830000
 C        1.3172540000       -3.0774870000       -0.7373500000
 C        0.7165190000       -1.9288860000       -0.1908820000
 C        2.8868660000       -0.8091710000       -0.3215770000
 C        1.5204940000       -0.8092760000        0.0019830000
 C       -0.7165060000       -1.9288840000        0.1908830000
 C       -1.3172420000       -3.0774810000        0.7373610000
 C       -2.6472210000       -3.0373060000        1.0811930000
 N       -3.4091100000       -1.9343780000        0.8915820000
 C       -2.8868500000       -0.8091660000        0.3215760000
 C       -1.5204800000       -0.8092740000       -0.0019900000
 H        3.1471130000       -3.8924860000       -1.5394370000
 H        0.7543150000       -3.9911510000       -0.9330820000
 H        1.1272380000        0.0891380000        0.4794630000
 H       -0.7543030000       -3.9911450000        0.9330960000
 H       -3.1471030000       -3.8924750000        1.5394480000
 H       -1.1272240000        0.0891360000       -0.4794760000
 C        4.7988900000       -1.9870630000       -1.3741760000
 H        5.1216080000       -0.9807560000       -1.6619940000
 H        5.4630100000       -2.3733300000       -0.5896180000
 H        4.8390240000       -2.6519000000       -2.2446910000
 C       -4.7988820000       -1.9870460000        1.3741620000
 H       -5.1216000000       -0.9807340000        1.6619650000
 H       -5.4629930000       -2.3733170000        0.5895980000
 H       -4.8390310000       -2.6518740000        2.2446820000
 C        3.7167280000        0.3671840000       -0.0432470000
 C       -3.7167190000        0.3671830000        0.0432430000
 C       -3.2651500000        1.6367550000        0.4449090000
 C       -4.9171550000        0.2757630000       -0.6856340000
 C       -5.6396210000        1.4200970000       -0.9886740000
 C       -4.0114280000        2.7710280000        0.1566440000
 C       -5.2126780000        2.6897160000       -0.5648200000
 H       -5.2697440000       -0.6871150000       -1.0621470000
 H       -6.5561820000        1.3307090000       -1.5771010000
 H       -2.3415220000        1.7343550000        1.0217160000
 H       -3.6569920000        3.7459740000        0.5000190000
 C       -6.0248630000        3.9098950000       -0.8579950000
 H       -5.4156890000        4.8232500000       -0.8292930000
 H       -6.8232040000        4.0231190000       -0.1048690000
 H       -6.5154570000        3.8429250000       -1.8392560000
 C        3.2651650000        1.6367470000       -0.4449490000
 C        4.9171380000        0.2757850000        0.6856750000
 C        5.6395900000        1.4201280000        0.9887160000
 C        5.2126580000        2.6897360000        0.5648200000
 C        4.0114300000        2.7710280000       -0.1566850000
 H        2.3415550000        1.7343320000       -1.0217860000
 H        3.6570020000        3.7459660000       -0.5000900000
 H        5.2697160000       -0.6870820000        1.0622260000
 H        6.5561300000        1.3307560000        1.5771770000
 C        6.0248290000        3.9099240000        0.8579960000
 H        6.5153930000        3.8429740000        1.8392730000
 H        5.4156510000        4.8232760000        0.8292600000
 H        6.8231910000        4.0231380000        0.1048920000


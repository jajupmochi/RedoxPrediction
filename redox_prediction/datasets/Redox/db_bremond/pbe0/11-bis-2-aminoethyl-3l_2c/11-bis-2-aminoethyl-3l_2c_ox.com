%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 11-bis-2-aminoethyl-3l_2c_ox

 3 2
 N       -3.5224240000       -0.4777920000       -0.1198500000
 C       -2.9116830000        0.4168310000       -0.9255460000
 C       -2.7890720000       -1.2810710000        0.6813130000
 C       -1.5314480000        0.5235630000       -0.9594410000
 C       -0.7406860000       -0.3188280000       -0.1613300000
 C       -1.4066450000       -1.2261060000        0.6782830000
 C        0.7439290000       -0.2757400000       -0.2285290000
 C        1.4388080000        0.9332210000       -0.3944750000
 C        2.8196830000        0.9313220000       -0.4965440000
 N        3.5217460000       -0.2204620000       -0.4419950000
 C        2.8825000000       -1.3973640000       -0.2664690000
 C        1.5047080000       -1.4534090000       -0.1523750000
 H        3.4995370000       -2.2990560000       -0.2423770000
 H        1.0411690000       -2.4369060000       -0.0488220000
 H        0.9235400000        1.8949980000       -0.4398400000
 H        3.3889940000        1.8517950000       -0.6413900000
 H       -3.3361960000       -1.9831770000        1.3154020000
 H       -0.8672600000       -1.8981930000        1.3491830000
 H       -3.5530640000        1.0270970000       -1.5645980000
 H       -1.0913730000        1.2393200000       -1.6571860000
 C       -4.9979760000       -0.5928660000       -0.1206980000
 C       -5.7113690000        0.4247310000        0.7881510000
 H       -5.2479550000       -1.6061250000        0.2199940000
 H       -5.3403450000       -0.5071780000       -1.1623420000
 N       -5.7274340000        1.7548700000        0.2847880000
 H       -5.2811990000        0.4160740000        1.8003600000
 H       -6.7507860000        0.0387650000        0.8855720000
 C        4.9946790000       -0.2001470000       -0.5872350000
 C        5.7521180000        0.1109620000        0.7164300000
 H        5.2441720000        0.5239080000       -1.3765550000
 H        5.3022210000       -1.1906710000       -0.9470930000
 N        5.6847940000        1.4715330000        1.1263320000
 H        5.4097700000       -0.5404120000        1.5338300000
 H        6.8071440000       -0.1753150000        0.5075830000
 H       -6.2899940000        1.9699470000       -0.5400920000
 H       -5.6332710000        2.5353250000        0.9344850000
 H        5.6190120000        1.6901180000        2.1203250000
 H        6.1738440000        2.1820400000        0.5794330000

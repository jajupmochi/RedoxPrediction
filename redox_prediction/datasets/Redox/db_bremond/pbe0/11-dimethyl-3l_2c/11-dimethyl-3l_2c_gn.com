%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT(calcall) FREQ SCF(tight) INT(ultrafine)

 11-dimethyl-3l_2c_gn

 2 1
 N        3.5209410000       -0.0074510000       -0.0019320000
 C        2.8497890000       -1.1048370000       -0.3973860000
 C        1.4650020000       -1.1323000000       -0.4033500000
 C        2.8527600000        1.0947780000        0.3947950000
 C        1.4708280000        1.1280880000        0.4019520000
 C        0.7421900000       -0.0017430000       -0.0008880000
 C       -0.7421900000        0.0017430000       -0.0009360000
 C       -1.4708540000       -1.1280820000        0.4018760000
 C       -1.4649760000        1.1322940000       -0.4034620000
 C       -2.8497630000        1.1048310000       -0.3975860000
 C       -2.8527860000       -1.0947720000        0.3946300000
 N       -3.5209410000        0.0074510000       -0.0021580000
 H       -3.4478290000        1.9619310000       -0.7139600000
 H       -0.9696520000        2.0403940000       -0.7527650000
 H       -0.9783020000       -2.0376360000        0.7513350000
 H       -3.4580210000       -1.9479380000        0.7100740000
 H        0.9782540000        2.0376480000        0.7513650000
 H        3.4579750000        1.9479490000        0.7102650000
 H        3.4478750000       -1.9619420000       -0.7137080000
 H        0.9697000000       -2.0404060000       -0.7526700000
 C        4.9947590000        0.0069600000        0.0052080000
 H        5.3686360000       -0.9585980000       -0.3514260000
 H        5.3497500000        0.1824540000        1.0298310000
 H        5.3521490000        0.8064850000       -0.6577510000
 C       -4.9947600000       -0.0069610000        0.0048860000
 H       -5.3686140000        0.9585970000       -0.3517710000
 H       -5.3498170000       -0.1824560000        1.0294860000
 H       -5.3521060000       -0.8064850000       -0.6580970000


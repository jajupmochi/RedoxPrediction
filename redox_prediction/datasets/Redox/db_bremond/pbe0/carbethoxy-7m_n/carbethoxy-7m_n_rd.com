%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 carbethoxy-7m_n_rd

-1 2
 C        0.8095670000        3.0484030000       -1.7507040000
 C       -0.3303040000        3.6305410000       -1.1503230000
 C        1.1317030000        1.7140200000       -1.4682100000
 C        0.3854440000        0.9818880000       -0.5570780000
 C       -0.7514880000        1.5824480000        0.0603900000
 C       -1.1094850000        2.9074360000       -0.2667930000
 H        1.4275550000        3.6252420000       -2.4428110000
 H        1.9842900000        1.2328560000       -1.9536680000
 H       -0.6017190000        4.6638380000       -1.3871350000
 H       -2.0038570000        3.3310030000        0.1959150000
 N        0.6151870000       -0.4057230000       -0.3033520000
 C       -0.5298140000       -1.2399710000       -0.4797800000
 C       -1.7410650000       -0.8242410000        0.1401980000
 S       -1.6710020000        0.5934420000        1.1544630000
 C       -0.5269370000       -2.3631170000       -1.2933560000
 C       -1.7018780000       -3.1000620000       -1.4900790000
 C       -2.9070780000       -2.6625570000       -0.9024980000
 C       -2.9332060000       -1.5328760000       -0.1030160000
 H        0.4104540000       -2.6693890000       -1.7623220000
 H       -1.6822550000       -4.0030810000       -2.1049390000
 H       -3.8549670000       -1.1702180000        0.3576300000
 H       -3.8310160000       -3.2230210000       -1.0743150000
 O       -3.0231980000        1.1538900000        1.3311590000
 O       -0.8905870000        0.3059580000        2.3694870000
 C        1.7519980000       -0.9129260000        0.2626240000
 O        2.6663180000        0.0570510000        0.4858480000
 O        1.9518560000       -2.0822950000        0.5089130000
 C        3.8685570000       -0.3582830000        1.1033410000
 C        4.8649030000       -0.9051100000        0.1017670000
 H        4.2609480000        0.5425640000        1.5991250000
 H        3.6446060000       -1.1153300000        1.8708420000
 H        4.4693590000       -1.8217060000       -0.3585960000
 H        5.8163760000       -1.1500050000        0.6004730000
 H        5.0693390000       -0.1681570000       -0.6903090000

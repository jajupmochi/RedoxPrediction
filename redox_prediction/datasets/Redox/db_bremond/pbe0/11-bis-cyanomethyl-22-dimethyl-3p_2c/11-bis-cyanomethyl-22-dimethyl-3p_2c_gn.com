%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 11-bis-cyanomethyl-22-dimethyl-3p_2c_gn

 2 1
 N        3.5150390000        0.0339590000       -0.0723970000
 C        2.8441560000       -1.0584230000        0.3541410000
 C        1.4659900000       -1.0885420000        0.3913360000
 C        0.7423310000        0.0414290000       -0.0123170000
 C        2.8595260000        1.1589940000       -0.4803340000
 C        1.4650960000        1.1579230000       -0.4424020000
 C       -0.7423320000        0.0414330000        0.0122870000
 C       -1.4659970000       -1.0885250000       -0.3913940000
 C       -2.8441620000       -1.0584010000       -0.3541940000
 N       -3.5150400000        0.0339730000        0.0723730000
 C       -2.8595210000        1.1589960000        0.4803340000
 C       -1.4650910000        1.1579190000        0.4424000000
 H        3.4501690000       -1.9119790000        0.6686690000
 H        0.9787280000       -1.9922420000        0.7619410000
 H        0.9538110000        2.0566660000       -0.7915940000
 H       -0.9787390000       -1.9922170000       -0.7620220000
 H       -3.4501800000       -1.9119480000       -0.6687370000
 H       -0.9538010000        2.0566530000        0.7916110000
 C        3.6334940000        2.3379430000       -0.9537270000
 H        4.2515500000        2.0920000000       -1.8326380000
 H        4.3081640000        2.7235030000       -0.1722330000
 H        2.9538430000        3.1485300000       -1.2401460000
 C       -3.6334820000        2.3379340000        0.9537670000
 H       -4.2514440000        2.0919850000        1.8327450000
 H       -4.3082400000        2.7234580000        0.1723340000
 H       -2.9538300000        3.1485460000        1.2401100000
 C        5.0018920000        0.0302610000       -0.1079710000
 C        5.5742280000       -1.2242050000        0.3482300000
 H        5.3711760000        0.8521410000        0.5258870000
 H        5.3292060000        0.2263600000       -1.1414630000
 N        6.0269430000       -2.2235900000        0.7117040000
 C       -5.0018930000        0.0302820000        0.1079480000
 C       -5.5742320000       -1.2242090000       -0.3481800000
 H       -5.3711740000        0.8521280000       -0.5259590000
 H       -5.3292060000        0.2264410000        1.1414270000
 N       -6.0269500000       -2.2236110000       -0.7116050000


%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 3i_2c_ox

 3 2
 C        3.6139430000       -0.0559120000       -0.4809150000
 C        3.0346000000       -1.3113370000       -0.4041890000
 C        2.8259070000        1.0281880000       -0.1343020000
 N        1.4905830000        0.9474660000        0.1447950000
 C        1.6443650000       -1.4163800000       -0.1403720000
 C        0.8337670000       -0.2361590000       -0.0072500000
 C        1.0922030000       -2.6912020000        0.1201060000
 C       -0.2650920000       -2.8283090000        0.4758830000
 C       -1.1429050000       -1.7312790000        0.3374030000
 C       -0.6381640000       -0.4450170000       -0.0523180000
 C       -2.5419770000       -1.9642940000        0.4028400000
 C       -3.4165990000       -0.9830700000       -0.0363600000
 C       -2.8696070000        0.1749760000       -0.5635050000
 N       -1.5318200000        0.4325460000       -0.5725680000
 C        1.0182620000        2.2024050000        0.8821860000
 C       -1.1534780000        1.7478820000       -1.1568100000
 C       -0.4443760000        2.4424870000        1.1498470000
 H        1.4479690000        3.0404650000        0.3130580000
 H        1.5651490000        2.1319840000        1.8377050000
 C       -1.2480910000        2.8190840000       -0.0819960000
 H       -1.8369230000        1.9321920000       -1.9975190000
 H       -0.1482660000        1.6503370000       -1.5882290000
 H       -0.4559800000        3.2866120000        1.8606090000
 H       -0.8967520000        1.6151600000        1.7246120000
 H       -2.3001290000        3.0024380000        0.1888600000
 H       -0.8834340000        3.7629160000       -0.5207080000
 H        3.6350400000       -2.2195800000       -0.5317130000
 H        4.6733610000        0.0916000000       -0.7136370000
 H        3.2677890000        2.0233030000       -0.0171390000
 H       -2.9150040000       -2.9328430000        0.7555180000
 H       -3.5006690000        0.9494560000       -1.0118170000
 H       -4.5026940000       -1.1198460000       -0.0281040000
 H        1.7392790000       -3.5761420000        0.0956580000
 H       -0.6566320000       -3.8105140000        0.7657650000

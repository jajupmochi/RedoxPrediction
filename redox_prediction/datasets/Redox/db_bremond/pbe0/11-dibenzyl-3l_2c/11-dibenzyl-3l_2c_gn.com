%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 11-dibenzyl-3l_2c_gn

 2 1
 N       -3.3422060000        1.1015440000       -0.3533530000
 C       -2.3602280000        1.9387900000       -0.7415210000
 C       -1.0385730000        1.5376100000       -0.7496740000
 C       -3.0559000000       -0.1507730000        0.0329030000
 C       -1.7481300000       -0.6130020000        0.0390790000
 C       -0.7043640000        0.2320850000       -0.3545690000
 C        0.7043630000       -0.2320630000       -0.3545830000
 C        1.7481290000        0.6129980000        0.0391200000
 C        1.0385730000       -1.5375630000       -0.7497750000
 C        2.3602270000       -1.9387420000       -0.7416470000
 C        3.0558990000        0.1507690000        0.0329140000
 N        3.3422060000       -1.1015230000       -0.3534240000
 H        2.6665720000       -2.9410600000       -1.0505010000
 H        0.2825180000       -2.2462100000       -1.0928850000
 H        1.5640390000        1.6329780000        0.3817280000
 H        3.9075100000        0.7643390000        0.3380640000
 H       -1.5640400000       -1.6330050000        0.3816200000
 H       -3.9075120000       -0.7643620000        0.3380120000
 H       -2.6665720000        2.9411270000       -1.0503090000
 H       -0.2825180000        2.2462800000       -1.0927370000
 C       -4.7634700000        1.6200630000       -0.3652920000
 C       -5.7882250000        0.6122870000        0.0398370000
 H       -4.7622710000        2.4894740000        0.3088160000
 H       -4.9331400000        1.9797270000       -1.3908190000
 C       -6.1730210000        0.4974130000        1.3833720000
 C       -7.1327730000       -0.4398180000        1.7584460000
 C       -7.7149810000       -1.2649000000        0.7955780000
 C       -6.3823080000       -0.2163220000       -0.9230050000
 C       -7.3415440000       -1.1523150000       -0.5442370000
 H       -6.1112320000       -0.1161980000       -1.9784100000
 H       -7.8125470000       -1.7851720000       -1.2995390000
 H       -8.4761960000       -1.9913250000        1.0894020000
 H       -5.7372770000        1.1578300000        2.1392550000
 H       -7.4401330000       -0.5156330000        2.8037190000
 C        4.7634690000       -1.6200410000       -0.3653950000
 C        5.7882240000       -0.6122890000        0.0397940000
 H        4.9331380000       -1.9796430000       -1.3909430000
 H        4.7622710000       -2.4894920000        0.3086610000
 C        6.1730240000       -0.4974980000        1.3833350000
 C        7.1327770000        0.4397090000        1.7584650000
 C        7.7149820000        1.2648510000        0.7956470000
 C        6.3823050000        0.2163790000       -0.9229980000
 C        7.3415430000        1.1523490000       -0.5441740000
 H        5.7372820000       -1.1579620000        2.1391790000
 H        7.4401390000        0.5154590000        2.8037420000
 H        8.4761980000        1.9912580000        1.0895150000
 H        6.1112270000        0.1163210000       -1.9784080000
 H        7.8125430000        1.7852530000       -1.2994380000

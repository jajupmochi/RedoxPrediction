%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 i47di-1-co2et-25-met-3-phen_n_ox

 1 2
 C       -1.1719040000        3.1124420000       -0.3132600000
 C        0.1554250000        3.3207710000       -0.1546290000
 C       -1.7963800000        1.7798550000       -0.3743570000
 C       -0.8652120000        0.6463230000       -0.1411150000
 C        0.4968940000        0.8156040000        0.0201060000
 C        1.1055140000        2.1710160000       -0.0056990000
 H       -1.8630590000        3.9501790000       -0.4397590000
 O       -2.9677570000        1.6095430000       -0.6314530000
 O        2.2996340000        2.3572670000        0.0897410000
 C        1.0970650000       -0.4955050000        0.0416000000
 C       -1.1205740000       -0.7465970000       -0.1605710000
 N        0.0619160000       -1.4061190000       -0.0779720000
 C       -2.4179610000       -1.4745670000       -0.3401030000
 O       -3.3322490000       -1.0412840000        0.4819860000
 O       -2.5161380000       -2.3391680000       -1.1719630000
 C       -4.6815800000       -1.5403560000        0.3095900000
 C       -5.5801860000       -0.7761840000        1.2405780000
 H       -4.9532350000       -1.4030180000       -0.7477020000
 H       -4.6720520000       -2.6217340000        0.5162500000
 H       -5.2897370000       -0.9256630000        2.2903550000
 H       -5.5514180000        0.2988890000        1.0137390000
 H       -6.6145800000       -1.1278660000        1.1187330000
 C        0.1613620000       -2.8589810000        0.0319610000
 H        0.1474330000       -3.3226020000       -0.9630470000
 H        1.0808700000       -3.1193900000        0.5661190000
 H       -0.6975820000       -3.2353250000        0.5990420000
 C        0.7797470000        4.6717780000       -0.1288190000
 H        1.5321200000        4.7618520000       -0.9278440000
 H        0.0292200000        5.4624980000       -0.2480040000
 H        1.3225130000        4.8273720000        0.8167290000
 C        2.4830950000       -0.8648010000        0.1025510000
 C        3.3745300000       -0.1420950000        0.9392740000
 C        2.9930400000       -1.9357450000       -0.6805280000
 C        4.3353740000       -2.2583250000       -0.6294940000
 C        5.1957180000       -1.5511950000        0.2245560000
 C        4.7095690000       -0.5015920000        1.0095980000
 H        2.9996250000        0.6751110000        1.5520060000
 H        5.3819830000        0.0413230000        1.6765270000
 H        4.7282260000       -3.0563810000       -1.2626780000
 H        6.2539950000       -1.8198410000        0.2701250000
 H        2.3431250000       -2.4645210000       -1.3787140000

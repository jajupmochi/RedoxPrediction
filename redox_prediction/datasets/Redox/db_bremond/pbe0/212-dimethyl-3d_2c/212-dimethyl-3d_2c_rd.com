%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 212-dimethyl-3d_2c_rd

 1 2
 C        3.5137390000       -0.1218170000       -0.1574850000
 C        2.8569330000       -1.3352530000        0.2072920000
 C        2.7733150000        0.9723380000       -0.4919280000
 N        1.4103130000        0.9620090000       -0.4802050000
 C        1.4779290000       -1.3213090000        0.2446740000
 C        0.7164560000       -0.1666910000       -0.0549390000
 C       -0.7164150000       -0.1667810000        0.0550860000
 N       -1.4103420000        0.9619170000        0.4802790000
 C       -1.4778610000       -1.3213800000       -0.2445590000
 C       -2.8568790000       -1.3353150000       -0.2073560000
 C       -3.5137270000       -0.1218240000        0.1571810000
 C       -2.7733350000        0.9723270000        0.4916880000
 C        0.6763940000        2.0864090000       -1.0547030000
 C       -0.0000870000        2.9489670000        0.0000930000
 H       -0.0701600000        1.6656910000       -1.7475270000
 H        1.3739320000        2.6775230000       -1.6619880000
 C       -0.6765410000        2.0863390000        1.0547920000
 H       -0.7297920000        3.6048970000       -0.4997830000
 H        0.7297770000        3.6047040000        0.5000210000
 H       -1.3740770000        2.6772800000        1.6622680000
 H        0.0700710000        1.6656690000        1.7475870000
 H       -0.9424400000       -2.2102360000       -0.5815430000
 H       -4.6023770000       -0.0563140000        0.1828720000
 H       -3.2337970000        1.9102540000        0.8076220000
 H        0.9425160000       -2.2101300000        0.5817620000
 H        4.6023850000       -0.0563420000       -0.1834640000
 H        3.2337650000        1.9102120000       -0.8080160000
 C       -3.6549010000       -2.5457700000       -0.5626180000
 H       -3.0159540000       -3.3965650000       -0.8323640000
 H       -4.3272210000       -2.3348640000       -1.4098550000
 H       -4.2951890000       -2.8478900000        0.2818520000
 C        3.6549950000       -2.5456530000        0.5626720000
 H        4.2955570000       -2.8476600000       -0.2816270000
 H        3.0160590000       -3.3965230000        0.8322090000
 H        4.3270610000       -2.3347110000        1.4101050000

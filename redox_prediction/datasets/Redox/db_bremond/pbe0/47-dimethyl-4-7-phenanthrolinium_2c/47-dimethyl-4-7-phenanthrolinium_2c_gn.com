%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 47-dimethyl-4-7-phenanthrolinium_2c_gn

 2 1
 C        3.4899690000        0.6167130000       -0.0000000000
 N        2.7912970000       -0.5219740000        0.0000000000
 C        1.4137650000       -0.5102270000        0.0000000000
 C        0.7259750000        0.7283920000       -0.0000000000
 C        2.8642790000        1.8607080000       -0.0000010000
 C        1.4862510000        1.9178210000       -0.0000010000
 C        0.6822380000       -1.7324450000        0.0000010000
 C       -0.6822380000       -1.7324450000        0.0000000000
 C       -1.4137650000       -0.5102270000       -0.0000000000
 C       -0.7259750000        0.7283920000       -0.0000000000
 N       -2.7912970000       -0.5219740000       -0.0000000000
 C       -1.4862510000        1.9178210000       -0.0000000000
 C       -3.4899690000        0.6167130000       -0.0000000000
 C       -2.8642790000        1.8607080000       -0.0000000000
 H        3.4756960000        2.7649650000       -0.0000010000
 H        1.0027630000        2.8956090000       -0.0000010000
 H        4.5782910000        0.5234750000       -0.0000000000
 H        1.2003320000       -2.6906570000        0.0000010000
 H       -1.2003320000       -2.6906570000        0.0000010000
 H       -1.0027630000        2.8956090000       -0.0000000000
 H       -3.4756960000        2.7649650000       -0.0000000000
 H       -4.5782910000        0.5234750000       -0.0000000000
 C        3.5252660000       -1.7987330000        0.0000010000
 H        4.5992320000       -1.5858200000        0.0000010000
 H        3.2777890000       -2.3735640000       -0.9023650000
 H        3.2777890000       -2.3735640000        0.9023660000
 C       -3.5252660000       -1.7987330000       -0.0000000000
 H       -3.2777880000       -2.3735650000       -0.9023650000
 H       -4.5992320000       -1.5858200000       -0.0000020000
 H       -3.2777900000       -2.3735640000        0.9023660000

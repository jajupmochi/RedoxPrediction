%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 3b_2c_gn

 2 1
 C        0.2726330000       -3.5200760000        0.0000000000
 C        1.5154550000       -2.8775680000        0.0000000000
 C        1.5694960000       -1.4893840000        0.0000000000
 C        0.3974650000       -0.7304930000        0.0000000000
 C       -0.8719210000       -2.7568820000        0.0000000000
 N       -0.8083680000       -1.3951040000        0.0000000000
 C        0.3974650000        0.7304940000       -0.0000000000
 N       -0.8083680000        1.3951040000       -0.0000000000
 C       -2.0010360000       -0.6689490000        0.0000000000
 C       -2.0010360000        0.6689490000       -0.0000000000
 C        1.5694960000        1.4893840000       -0.0000000000
 C        1.5154550000        2.8775680000       -0.0000000000
 C       -0.8719210000        2.7568830000       -0.0000000000
 C        0.2726330000        3.5200760000       -0.0000000000
 H       -2.9292070000       -1.2420140000        0.0000000000
 H       -2.9292070000        1.2420140000       -0.0000000000
 H        0.1886560000       -4.6095720000        0.0000000000
 H       -1.8744540000       -3.1901220000        0.0000000000
 H        2.4410160000       -3.4596880000        0.0000000000
 H        2.5400030000       -0.9949120000        0.0000000000
 H       -1.8744550000        3.1901220000       -0.0000000000
 H        0.1886560000        4.6095720000       -0.0000000000
 H        2.5400030000        0.9949120000       -0.0000000000
 H        2.4410160000        3.4596880000       -0.0000000000


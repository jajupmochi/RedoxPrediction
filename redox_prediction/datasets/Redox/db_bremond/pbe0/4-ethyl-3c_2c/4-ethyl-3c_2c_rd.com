%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 4-ethyl-3c_2c_rd

 1 2
 C       -4.1643120000       -0.3690800000       -0.0982450000
 C       -3.8062510000        0.9955590000       -0.2585450000
 C       -2.4860720000        1.3639530000       -0.2128840000
 C       -3.1704920000       -1.2799260000        0.1104460000
 N       -1.8602940000       -0.9103090000        0.1643350000
 C       -1.4589670000        0.4065290000       -0.0060130000
 C       -0.0703740000        0.7398290000        0.0342920000
 N        0.8958030000       -0.2597500000       -0.0602780000
 C       -0.8354050000       -1.8796490000        0.4986080000
 C        0.4082980000       -1.6086000000       -0.3172430000
 H       -1.2113200000       -2.8887720000        0.2847530000
 H       -0.6061820000       -1.8202050000        1.5769060000
 H        0.1917220000       -1.7231120000       -1.3939140000
 H        1.1755000000       -2.3401140000       -0.0484850000
 C        0.3759110000        2.0749790000        0.1683670000
 C        1.7144030000        2.3631700000        0.2112890000
 C        2.6604300000        1.3183120000        0.1156730000
 C        2.2476620000        0.0152610000       -0.0241940000
 H       -5.2018910000       -0.6998110000       -0.1429710000
 H       -3.3657650000       -2.3447950000        0.2503770000
 H        3.7256190000        1.5371270000        0.1509730000
 H       -2.2129670000        2.4058240000       -0.3730500000
 H       -0.3570140000        2.8730170000        0.2730320000
 H       -4.5765910000        1.7486270000       -0.4339100000
 H        2.0499100000        3.3950040000        0.3301260000
 C        3.2154370000       -1.1289840000       -0.1542540000
 C        4.6866510000       -0.7501210000       -0.1317810000
 H        2.9965980000       -1.6701560000       -1.0916910000
 H        3.0196700000       -1.8501560000        0.6599900000
 H        4.9493470000       -0.0764300000       -0.9604260000
 H        5.3001890000       -1.6552170000       -0.2373400000
 H        4.9730990000       -0.2678040000        0.8141350000


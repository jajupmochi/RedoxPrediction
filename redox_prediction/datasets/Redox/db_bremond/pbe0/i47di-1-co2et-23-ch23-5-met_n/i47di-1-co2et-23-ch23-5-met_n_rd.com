%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT FREQ SCF(tight) INT(ultrafine)

 i47di-1-co2et-23-ch23-5-met_n_rd

-1 2
 C        2.8288900000       -1.8439200000        0.0000720000
 C        3.5491720000       -0.6618940000        0.0336850000
 C        1.3854990000       -1.9934570000       -0.0245220000
 C        0.6708950000       -0.7113710000       -0.0262400000
 C        2.8739640000        0.6163880000        0.0490150000
 C        1.4220610000        0.5080240000        0.0189420000
 C       -0.7077260000       -0.3572430000       -0.0469980000
 C        0.5022560000        1.5465060000        0.0206570000
 N       -0.7454500000        1.0350330000       -0.0249020000
 O        0.8442700000       -3.1159900000       -0.0359860000
 O        3.4645300000        1.7210310000        0.0829020000
 C       -1.7725660000        2.0594070000        0.0046100000
 C       -0.9769960000        3.3354320000       -0.3205250000
 H       -2.2469900000        2.0967360000        1.0012370000
 H       -2.5682560000        1.8462440000       -0.7196790000
 C        0.4722840000        3.0319430000        0.1009780000
 H       -1.4018320000        4.2229870000        0.1724130000
 H       -1.0120320000        3.5148780000       -1.4071480000
 H        0.6828830000        3.3643250000        1.1338830000
 H        1.2399770000        3.4965130000       -0.5328810000
 C       -1.9070130000       -1.1673480000       -0.0790440000
 O       -2.0109630000       -2.3627860000       -0.2047980000
 O       -3.0426970000       -0.3878030000        0.0660040000
 C       -4.2579900000       -1.1001710000        0.0454180000
 C       -5.3892340000       -0.1136900000        0.2193920000
 H       -4.2613310000       -1.8608970000        0.8452320000
 H       -4.3528080000       -1.6601700000       -0.9013890000
 H       -5.2901360000        0.4333970000        1.1695930000
 H       -6.3584720000       -0.6360690000        0.2219340000
 H       -5.3998840000        0.6249160000       -0.5971920000
 C        5.0469890000       -0.6325950000        0.0594430000
 H        5.4730500000       -1.6475710000        0.0508950000
 H        5.4172450000       -0.0993790000        0.9519790000
 H        5.4471250000       -0.0727430000       -0.8034620000
 H        3.3655850000       -2.8000890000       -0.0053860000


%nproc=8
%mem=18000mb
#P PBE1PBE DEF2TZVPP OPT(calcall) FREQ SCF(tight) INT(ultrafine)

 5-apz-14-dione_n_gn

 0 1
 C       -2.2553010000        1.2518540000        0.0004570000
 C       -1.3802420000        2.3220680000        0.0007730000
 C       -1.7849520000       -0.0856530000       -0.0001680000
 C       -0.3739770000       -0.2827850000       -0.0000870000
 C        0.4926930000        0.8292300000        0.0001400000
 C        0.0080530000        2.1281780000        0.0004930000
 C        0.2059990000       -1.6093310000       -0.0000430000
 N        1.6629720000       -1.7652810000        0.0016620000
 C        1.9457810000        0.5813460000       -0.0004580000
 N        2.4350700000       -0.7994760000        0.0015850000
 O        2.7829120000        1.4438040000       -0.0021780000
 O       -0.4219580000       -2.6502840000       -0.0009010000
 N       -2.6344220000       -1.1271920000       -0.0008840000
 H       -3.6315200000       -0.9758770000       -0.0000300000
 H       -2.2425370000       -2.0659610000       -0.0008490000
 H       -3.3341190000        1.4284080000        0.0005780000
 H       -1.7844430000        3.3374180000        0.0011980000
 H        0.7113150000        2.9620580000        0.0005490000

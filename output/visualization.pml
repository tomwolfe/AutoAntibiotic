# Auto-generated PyMOL visualization script (AutoAntibiotic)
# Load with: pymol -l visualization.pml

load output/workdir/PBP2a_clean.pdb, PBP2a
select conserved_residues, (resn SER and resi 403) or (resn LYS and resi 406) or (resn TYR and resi 446)
show sticks, conserved_residues
color magenta, conserved_residues

load output/workdir/ALL_QU05_active_c2_out.pdbqt, Ligand_1_ALL_QU05
util.cbaw Ligand_1_ALL_QU05
show sticks, Ligand_1_ALL_QU05
distance hbond_ser_1, Ligand_1_ALL_QU05, resi 403 and name OG, cutoff=3.5
distance hbond_lys_1, Ligand_1_ALL_QU05, resi 406 and name NZ, cutoff=3.8
distance hbond_tyr_1, Ligand_1_ALL_QU05, resi 446 and name OH, cutoff=3.5

load output/workdir/BRICS_01163_active_c2_out.pdbqt, Ligand_2_BRICS_01163
util.cbaw Ligand_2_BRICS_01163
show sticks, Ligand_2_BRICS_01163
distance hbond_ser_2, Ligand_2_BRICS_01163, resi 403 and name OG, cutoff=3.5
distance hbond_lys_2, Ligand_2_BRICS_01163, resi 406 and name NZ, cutoff=3.8

load output/workdir/BRICS_0022_active_c2_out.pdbqt, Ligand_3_BRICS_0022
util.cbaw Ligand_3_BRICS_0022
show sticks, Ligand_3_BRICS_0022
distance hbond_ser_3, Ligand_3_BRICS_0022, resi 403 and name OG, cutoff=3.5
distance hbond_lys_3, Ligand_3_BRICS_0022, resi 406 and name NZ, cutoff=3.8
distance hbond_tyr_3, Ligand_3_BRICS_0022, resi 446 and name OH, cutoff=3.5


# Rendering settings
set dash_width, 2
set bg_rgb, white
set ray_opaque_background, 1

# Orient the view
zoom
orient

# Save image
png output/visualization.png, dpi=300

# Auto-generated PyMOL visualization script (AutoAntibiotic)
# Load with: pymol -l visualization.pml

load 'output/workdir/PBP2a_clean.pdb', PBP2a
select conserved_residues, (resn SER and resi 403) or (resn LYS and resi 406) or (resn TYR and resi 446)
show sticks, conserved_residues
color magenta, conserved_residues

load 'output/workdir/ALL_QU05_active_c2_out.pdbqt', Ligand_1_ALL_QU05
color byelement, Ligand_1_ALL_QU05
show sticks, Ligand_1_ALL_QU05
distance hbond_ser_1, Ligand_1_ALL_QU05, resi 403 & name OG, cutoff=3.5
dash wid 2.0
distance hbond_lys_1, Ligand_1_ALL_QU05, resi 406 & name NZ, cutoff=3.8
dash wid 2.0
distance hbond_tyr_1, Ligand_1_ALL_QU05, resi 446 & name OH, cutoff=3.5
dash wid 2.0

load 'output/workdir/BRICS-01163_active_c2_out.pdbqt', Ligand_2_BRICS-01163
color byelement, Ligand_2_BRICS-01163
show sticks, Ligand_2_BRICS-01163
distance hbond_ser_2, Ligand_2_BRICS-01163, resi 403 & name OG, cutoff=3.5
dash wid 2.0
distance hbond_lys_2, Ligand_2_BRICS-01163, resi 406 & name NZ, cutoff=3.8
dash wid 2.0

load 'output/workdir/SEED-01150_active_c2_out.pdbqt', Ligand_3_SEED-01150
color byelement, Ligand_3_SEED-01150
show sticks, Ligand_3_SEED-01150
distance hbond_ser_3, Ligand_3_SEED-01150, resi 403 & name OG, cutoff=3.5
dash wid 2.0
distance hbond_lys_3, Ligand_3_SEED-01150, resi 406 & name NZ, cutoff=3.8
dash wid 2.0
distance hbond_tyr_3, Ligand_3_SEED-01150, resi 446 & name OH, cutoff=3.5
dash wid 2.0


# Orient the view
zoom
orient

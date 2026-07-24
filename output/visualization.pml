# Auto-generated PyMOL visualization script (AutoAntibiotic)
# Load with: pymol -l visualization.pml

load 'output/workdir/PBP2a_clean.pdb', PBP2a
select conserved_residues, (resn SER and resi 403) or (resn LYS and resi 406) or (resn TYR and resi 446)
show sticks, conserved_residues
color magenta, conserved_residues

load 'output/workdir/AA-0039_active_c2_out.pdbqt', Ligand_1_AA-0039
color byelement, Ligand_1_AA-0039
show sticks, Ligand_1_AA-0039
distance hbond_lys_1, Ligand_1_AA-0039, resi 406 & name NZ, cutoff=3.8
dash wid 2.0
distance hbond_tyr_1, Ligand_1_AA-0039, resi 446 & name OH, cutoff=3.5
dash wid 2.0

load 'output/workdir/AA-0033_active_c2_out.pdbqt', Ligand_2_AA-0033
color byelement, Ligand_2_AA-0033
show sticks, Ligand_2_AA-0033
distance hbond_tyr_2, Ligand_2_AA-0033, resi 446 & name OH, cutoff=3.5
dash wid 2.0

load 'output/workdir/AA-0103_active_c2_out.pdbqt', Ligand_3_AA-0103
color byelement, Ligand_3_AA-0103
show sticks, Ligand_3_AA-0103
distance hbond_lys_3, Ligand_3_AA-0103, resi 406 & name NZ, cutoff=3.8
dash wid 2.0
distance hbond_tyr_3, Ligand_3_AA-0103, resi 446 & name OH, cutoff=3.5
dash wid 2.0


# Orient the view
zoom
orient

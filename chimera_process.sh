for dataset in general-set-except-refined refined-set; do
    echo $dataset
    for pdbfile in database/$dataset/*/*_pocket.pdb; do
        mol2file=${pdbfile%pdb}mol2
        if [[ ! -e $mol2file ]]; then
            echo -e "open $pdbfile \n addh \n addcharge \n write format mol2 0 tmp.mol2 \n stop" | chimera --nogui
            sed 's/H\.t3p/H    /' tmp.mol2 | sed 's/O\.t3p/O\.3  /' > $mol2file
        fi
    done
done > chimera_rw.log

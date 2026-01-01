

This repo contians the Notebooks + plain Python scripts from my B.Sc. thesis work on Physics-Informed Neural Networks (PINNs) for neutronic analysis.

## What's inside
- `notebooks/` : original notebooks (best to skim)
- `scripts/`   : plain-python versions 

## Quick run
```bash
pip install -r requirements.txt
python scripts/01_1D_line_source.py
python scripts/02_dis_1D.py
python scripts/03_lcrm_compare.py
python scripts/04_lcrm_main.py
```

Notes:
- Some scripts generate lots of plots during training for intermediate checking of the state of the network
- On CPU, training may take a while, therefore running on a CUDA enable GPU is preferred. 

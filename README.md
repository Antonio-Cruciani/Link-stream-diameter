# Temporal graphs diameter
## Usage
All files to generate stats are in folder `./src/stats/`. 

### Input
The statistics will be generated on the input graphs. All graphs must be sorted in increasing order of times. All the graphs paths you want to generate statistics must be in file `./src/stats/TGraphs/TGraphs_list` that you should create.

Each line of that file should be respect this form:\
`<path of the graph> [dummy] [0]` : where dummy is an optional integer value that represents the last not dummy node in the graph. You can use a negative value like -1 if there aren't dummy nodes in the graph. The last value of the line ([0]) is optional and if there is, the graph is undirected.\
You can use the simplest form `<path of the graph>` if it is DiGraph and there aren't dummy nodes in the graph.

### Generate stats
- `Diameter FP and SP:` run file `./stats/diameter_stats.py` with appropriate args. For more info read the help: `./stats/diameter_stats.py --help`
- `Diameter EAT and LDT:` run file `./stats/rub_diameter_stats.py` with appropriate args. For more info read the help: `./rub_diameter_stats.py --help`
- `Double sweep:` run file `./stats/two_sweep_stats.py` with appropriate args for double sweep EAT, FP or SP. For more info read the help: `./stats/two_sweep_stats.py --help`. Run `./stats/two_sweep_ldt_stats.py` for double sweep LDT.
- `iFUB:` Run `iFUB_eat_stats.py` for iFUB EAT, `./stats/iFUB_ldt_stats.py ` for iFUB LDT, `./stats/iFUB_fp_stats.py ` for iFUB FP, `./stats/iFUB_sp_stats.py `for iFUB SP. Like before, for more info use `-- help`.

# Link stream diameter
## Usage
All files needed to generate stats are in the folder `./src/stats/`. 

### Dataset

The link at the dataset is here: 
Each link stream is a list of temporal edges u v t lambda (see the paper). The links specified in each link stream must be sorted in increasing order of t.

### Generate stats
The list of paths of the link streams to be considered must be written in a input file that must be passed in input.
We provide the following files, FileA and FileB, which allow to run our experiments on the dataset we used and that can be downloaded using the link below.\\

FileA are link stream with dummy\
FileB are link stream weighted\\

Each line of the input file, as it can be seen in FileA and FileB, respect this format:\
`<path of the link stream> [dummy] [0]` : where `dummy` is an optional integer value that represents the last not dummy node in the graph, as we assume that all dummy nodes have index greater or equal than `dummy`. We use a negative value like -1 if there aren't dummy nodes in the graph. This is to deal with the transformations described in Section 1.2 of the paper. The last value of the line ([0]) is optional and if there is, the graph is undirected.\
We also use the simplest form `<path of the link stream>` if it is a directed link stream and there aren't dummy nodes in the graph.

##### Computing Lower bounds
`cd ./src/stats/`
- EAT: run `Python3 two_sweep_stats.py EAT FileA`. 
- LDT: run `Python3 two_sweep_ldt_stats.py FileA`. 
- FT: run `Python3 two_sweep_stats.py FT FileA`.
- ST: run `Python3 two_sweep_stats.py FT FileB`.

##### Computing the exact diameter for EAT and LDT using our algorithm:
`cd ./src/stats/`
- EAT: run `Python3 rub_diameter_stats.py EAT FileA` 
- LDT: run `Python3 rub_diameter_stats.py LDT FileA`

##### Computing the exact diameter for FT and ST using the text-book algorithm:
`cd ./src/stats/`
- FT: run `Python3 tb_diameter_stats.py FT FileA`
- ST: run `Python3 tb_diameter_stats.py ST FileB`

###### Computing pivot diameter
`cd ./src/stats/`
- EAT: run `Python3 iFUB_eat_stats.py NumHubs NumTimes FileA`. 
- LDT: run `Python3 iFUB_ldt_stats.py NumHubs NumTimes FileA`. 
- FT: run `Python3 iFUB_ft_stats.py NumHubs NumTimes FileA`.
- ST: run `Python3 iFUB_st_stats.py NumHubs NumTimes FileB`.


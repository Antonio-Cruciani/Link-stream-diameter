# Link stream diameter
Code associated with the paper: *M. Calamai, P. Crescenzi, A. Marino "On Computing the Diameter of (Weighted) Link Streams"*.
## Usage
All files needed to generate stats are in the folder `./src/stats/`. 

### Dataset

The link at the dataset is here: https://bit.ly/3jv1AZh \
Once unzipped, in order to use the facilities we provide below, the folder `graphs` must be placed in the folder `./src/stats/`.
Each file in the folder `graphs` is a link stream, i.e. a list of temporal edges u v t lambda (see the paper). The links specified in each link stream are (and must be) sorted in increasing order of t.

### Generate stats
The list of paths of the link streams to be considered must be written in a input file that must be passed in input.
We provide the following files, fileA and fileB, which allow to run our experiments on the dataset we used and that can be downloaded using the link below.

fileA: https://bit.ly/2YTyEAw \
fileB: https://bit.ly/2MEsbXQ

The file fileA and fileB assume that the folder `graphs` is placed in the folder `./src/stats/`. Each line of the input file, as it can be seen in fileA and fileB, respect this format:
`<path of the link stream> <[dummy]> <[0]>` : where `dummy` is an optional integer value that represents the last not dummy node in the graph, as we assume that all dummy nodes have index greater or equal than `dummy`. We use a negative value like -1 if there aren't dummy nodes in the graph. This is to deal with the transformations described in Section 1.2 of the paper. The last value of the line ([0]) is optional and if there is, the graph is undirected.\
We also use the simplest form `<path of the link stream>` if it is a directed link stream and there aren't dummy nodes in the graph.

##### Computing Lower bounds:
`cd ./src/stats/`
- EAT: run `python3 double_sweep_stats.py EAT fileA`
- LDT: run `python3 double_sweep_stats.py LDT fileA`
- FT: run `python3 double_sweep_stats.py FT fileA`
- ST: run `python3 double_sweep_stats.py ST fileB`

##### Computing the exact diameter for EAT and LDT using our algorithm:
`cd ./src/stats/`
- EAT: run `python3 rub_diameter_stats.py EAT fileA`
- LDT: run `python3 rub_diameter_stats.py LDT fileA`

##### Computing the exact diameter using the text-book algorithm:
`cd ./src/stats/`
- EAT: run `python3 text_book_diameter_stats.py EAT fileA`
- LDT: run `python3 text_book_diameter_stats.py LDT fileA`
- FT: run `python3 text_book_diameter_stats.py FT fileA`
- ST: run `python3 text_book_diameter_stats.py ST fileB`

##### Computing pivot diameter:
`cd ./src/stats/`
- EAT: run `python3 pivot_ifub_diameter_stats.py EAT NumHubs NumTimes fileA`
- LDT: run `python3 pivot_ifub_diameter_stats.py LDT NumHubs NumTimes fileA`
- FT: run `python3 pivot_ifub_diameter_stats.py FT NumHubs NumTimes fileA`
- ST: run `python3 pivot_ifub_diameter_stats.py ST NumHubs NumTimes fileB`


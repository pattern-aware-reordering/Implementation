# How to run the pattern precision comparison?

0. Make sure that your computer has installed matlab, pycharm, and python3.

1. run `./synthetic_generator.py` to generate synthetic datasets. They will be stored in `./data/synthetic-x.x-x.x-x.x.json`, `./data/synthetic-x.x-x.x-x.x.edgelist`, and `./data/synthetic-x.x-x.x-x.x.out`.

2. run `./VoG/run_structureDiscovery.m` to run VoG. The detected patterns are summarized in `./VoG/DATA/synthetic-x.x-x.x-x.x_ALL.model`.

3. run `./pattern_precision.py` to run our model. It will store the results of our model and VoG into a unified file: `./precision.csv`

4. run `./pattern_significance.py` to run the Conover's test. It will generate the p-value.

5. run `./draw-precision.html` (require a localhost) to see the line charts.

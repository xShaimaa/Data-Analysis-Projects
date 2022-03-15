# Auto-MPG Dataset Analysis

| Contents 											 	   	|
| -------- 											 	   	|
| [Dataset Description](#Dataset-Description)			   	|
| [Columns Descreption](#Columns-Descreption) 		   		|
| [Data Wrangling](#Data-Wrangling)					   		|
| [Data Cleaning](#Data-Cleaning)						   	|
| [Data Visualization](#Data-Visualization)					|
| [Conclusion](#Conclusion)									|
| [Built with](#Built-with)							   		|

## Dataset Description: 
The MPG dataset is technical spec of cars originaly provided from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/auto+mpg) and can be found on Kaggle [here](https://www.kaggle.com/uciml/autompg-dataset). 
The data concerns city-cycle fuel consumption in miles per gallon to be analyzed in terms of 3 multivalued discrete and 5 continuous attributes.

## Columns Descreption:
1. `mpg`: miles per galon of fuel (continuous variable).
2. `cylinders`: number of engine cylinders (multi-valued discrete variable).
3. `displacement`: (continuous variable)
4. `horsepower`: the power produced by engine to move the car (continuous variable)
5. `weight`: car weight (continuous variable)
6. `acceleration`: the acceleration an engine can get per second (continuous variable)
7. `model year`: car release year from 1970 to 1982(multi-valued discrete variable)
8. `origin`: car manufacturing place (1 -> USA, 2 -> Europe, 3 -> Asia) (multi-valued discrete variable)
9. `car name`: car model name (unique for each instance)

## Data Wrangling:
Our data can be found on `auto-mpg.csv` file provided on this repository, downloaded from [Kaggle](https://www.kaggle.com/uciml/autompg-dataset). 

## Data Cleaning:
**Exploring Summary**
1. Our dataset had a total of 398 records and 9 columns.
2. We had no NaNs in our dataset nor duplicated rows.
3. `horsepower` column had inconsistant data type that needed to be handled and casted to `int`.
4. `origin` needed to be parsed and casted into a categorical datatype.
5. No columns needed to be dropped.

## Data Visualization
Using `Matplotlib` and `Seaborn`, we made several meaningful visuals and charts to help us gain informative insights regarding any correlation between attributes in our dataset, that'll be discussed in the next section.

## Conclusion
These are derived conclusions after comleting our data visualisation phase.
1. As years pass after `1973`, there has been a noticable increase in `mpg`.
2. As `cylinders` in the engine increases above 4, `MPG` decreases and engine `horsepower` increases. That indicates negative correlation between `mpg` and `horsepower`.
3. `mpg` increases as `weight` decreses over time, that also indecates a stron correlation between them.
4. Althogh `USA` has the biggest count of produced cars, its cars has relatively very low `mpg`, thus the highest possible `weight` compared to `Asia` and `Europe`
5. `Asia` is the leading contry in producing cars with high `mpg` with a mean close to 30, and it produces the lightest cars
6. Wa can spot a negative correlation between `acceleration` and `horepower`, this means that it has a positive one with `mpg`.

## Built with:
| Tools 		|
| -------- 		|
| JupyterLab	|
| Python3	   	|
| Pandas		|
| Numpy			|
| Matplotlib	|
| Seaborn		|

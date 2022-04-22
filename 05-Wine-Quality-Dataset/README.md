# **Wine Quality Dataset Analysis and EDA**

| Contents 											 	   	|
| -------- 											 	   	|
| [Dataset Description](#Dataset-Description)			   	|
| [Columns Descreption](#Columns-Descreption) 		   		|
| [EDA Questions](#eda-questions)							|
| [Data Wrangling](#Data-Wrangling)					   		|
| [Data Cleaning](#Data-Cleaning)						   	|
| [Data Visualization](#Data-Visualization)					|
| [Conclusion](#Conclusion)									|
| [Built with](#Built-with)							   		|

## Dataset Description: 
There are two datasets that provide information on samples of red and white variants of the Portuguese "Vinho Verde" wine. 
Each sample of wine was rated for quality by wine experts and examined with physicochemical tests. Due to privacy and logistic issues, 
only data on these physicochemical properties and quality ratings are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.). 
data is originaly from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality).


## Columns Descreption:
1. `fixed acidity`: most acids involved with wine or fixed or nonvolatile (do not evaporate readily)
2. `volatile acidity`: the amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste
3. `citric acid`: found in small quantities, citric acid can add 'freshness' and flavor to wines
4. `residual sugar`: the amount of sugar remaining after fermentation stops, it's rare to find wines with less than 1 gram/liter and wines with greater than 45 grams/liter are considered sweet
5. `chlorides`: the amount of salt in the wine
6. `free sulfur dioxide`: the free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) and bisulfite ion; it prevents microbial growth and the oxidation of wine
7. `total sulfur dioxide`: amount of free and bound forms of S02; in low concentrations, SO2 is mostly undetectable in wine, but at free SO2 concentrations over 50 ppm, SO2 becomes evident in the nose and taste of wine
8. `density`: the density of water is close to that of water depending on the percent alcohol and sugar content
9. `pH`: describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic); most wines are between 3-4 on the pH scale
10. `sulphates`: a wine additive which can contribute to sulfur dioxide gas (S02) levels, wich acts as an antimicrobial and antioxidant
11. `alcohol`: the percent alcohol content of the wine
12. `quality`: (score between 0 and 10)


## EDA Questions:
- Q1: What chemical characteristics are most important in predicting the quality of wine?
- Q2: Is a certain type of wine (red or white) associated with higher quality?
- Q3: Do wines with higher alcoholic content receive better ratings?
- Q4: Do sweeter wines (more residual sugar) receive better ratings?
- Q5: What level of acidity (pH) is associated with the highest quality?


## Data Wrangling:
Our data can be found on `wineQualityReds.csv` and `wineQualityWhites.csv` files provided on this repository, 
downloaded from [Kaggle](https://www.kaggle.com/datasets/danielpanizzo/wine-quality) 
and originaly from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wine+Quality). 


## Data Cleaning:
### Exploration Summery
- red dataframe consists of 1599 records and 13 attributes, while white dataframe consists of 4898 records and the same attributes.
- both data frames has no NaNs nor duplicated values.
- we woul combine the two dataframes and append a new categorical column to indecate the wine color for better analysis.
- columns data types are consistant.
- `Unnamed: 0` column would be dropped.

We endded up with with 13 columns and 6497 rows for our data to begin the analysis with. 
a new csv file containing our full data is saved in `wine_full.csv`.


## Data Visualization
Using `Matplotlib` and `Seaborn`, we made several meaningful visuals and charts to help us gain informative insights regarding any correlation between attributes in our dataset, that'll be discussed in the next section.


## Conclusion
These are derived conclusions after completing our data visualisation phase.

### Q1: What chemical characteristics are most important in predicting the quality of wine?
- the vast majority of the wine has a `quality` of 6, while less numbers has a `quality` of 9.
- using correlation plot, we can easily see if certain attributes are correlated more strongly to wine `quality` than some others.

  - strong correlated attributes:
    - `alcohol` and `quality`, and it's clear that this is the highest relation that affects wine `quality`.
  - weak correlated attributes (do not depend on each other):
    - `density` and `alcohol`.
    - `free.sulphur.dioxide` and `citric.acid` has almost no correlation with quality
  - `density` has strong positive correlation with `residual.sugar` and strong negative correlation with `alcohol`.

---
### Q2: Is a certain type of wine (red or white) associated with higher quality?
- there is noticable deviation between `white` and `red` wine counts.
- `white` wine formes the vast majority of our dataset as it appears in more than 75% of the times.
- most of the `white` wine has a `quality` of 6, while most of the `red` wine has a `quality` of 5.
- the mean `quality` of `red` and `white` wine are ve`ry close.
- `white` wine has the best mean `quality` higher than `red` wine.

---
### Q3: Do wines with higher alcoholic content receive better ratings?
- we have the highst `alcohol` content at 14.9.
- most of the wine has `alcoholic` content around 10.4.
- most of our dataset that has a `quality` of 6 appears to have relatively low `acoholic` content, but it's still above the mean.
- high `alcoholic` content only appears in our dataset with high `quality` wine.

---
### Q4: Do sweeter wines (more residual sugar) receive better ratings?
- we can see that the highest `sugar` content is tied to a `quality` of 5, while lower `sugar` content appears to have respectively higher `quality`.

---
### Q5: What level of acidity (pH) is associated with the highest quality?
- most of the wine in our dataset has high `acidity level`
- it's clear that all four acidity levels has close mean `quality`, but the `Low acidity` level has the highest `quality` in our dataset.


## Built with:		
- JupyterLab	
- Python3	   	
- Pandas		
- Numpy			
- Matplotlib	
- Seaborn		

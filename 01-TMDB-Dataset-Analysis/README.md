# TMDB Movies Dataset Analysis 
### Udacity Become a Data Analyst Nanodegree | Project 2

| Contents 											 	   	|
| -------- 											 	   	|
| [Dataset Description](#Dataset-Description)			   	|
| [Columns Descreption](#Columns-Descreption) 		   		|
| [Questions for Analysis](#Questions-for-Analysis)	   		|
| [Data Wrangling](#Data-Wrangling)					   		|
| [Data Cleaning](#Data-Cleaning)						   	|
| [Exploratory Data Analysis](#Exploratory-Data-Analysis)	|
| [Built with](#Built-with)							   		|

## Dataset Description: 
This data set contains information about 10,000 movies extracted from [TMDB](https://www.themoviedb.org/). The dataset contains movies from 1960 to 2015. Including user ratings and revenue. Original data from [Kaggle](https://www.kaggle.com/tmdb/tmdb-movie-metadata)

## Columns Descreption:
- `id, imdb_id`: unique id or imdb id for each movie on TMDB
- `popularity`: a metric used to measure the popularity of the movie.
- `budget`:the total budget of the moviein USD.
- `revenue`:the total revenue of the movie in USD.
- `original_title`: the original title of the movie.
- `cast`:the names of the cast of the movie separated by "|".
- `homepage`: the website of the movie (if it existed).
- `director`:name(s) of the director(s) of the movie (separated by "|" if there are more than one director).
- `tagline`:a catchphrase describing the movie.
- `keywords`: keywords related to the movie.
- `overview`:summary of the plot of the movie.
- `runtime`:total runtime of the movie in minutes.
- `genres`: genres of the movie separated by "|".
- `production_companies`:production compan(y/ies) of the movie.
- `release_date`:release date of the movie.
- `vote_count`:number of voters of te movie.
- `vote_average`:the average user rating of the movie
- `release_year`:release year of the movie (from 1960 to 2015)
- `budget_adj`:the total budget of the moviein USD in terms of 2010 dollars, accounting for inflation over time.
- `revenue_adj`:the total budget of the movie in USD in terms of 2010 dollars, accounting for inflation over time.

## Questions for Analysis:
- Do movies with high popularity achive high revenvue?
- What are the most filmed genres in this whole dataset?
- Is there a correlation between a movie budget and its revenue?

## Data Wrangling:
Our data can be found on `tmdb-movies.csv` file provided on this repository. It is an edited version of the original Kaggle's [TMDB 5000 Movie Dataset](https://www.kaggle.com/tmdb/tmdb-movie-metadata) provided by Udacity on the Become a Data Analyst Nanodegree Program. 

## Data Cleaning:
**Main Observations:**
1. Our dataset consisted of a total of 10866 rows and 21 columns.
2. We had only 1 duplicated row which had been dropped.
3. Some columns wont be useful in answering our questions so they were dropped.
4. Few columns had many missing values that needed to be handled.
5. Columns `cast` `director` `genre` had values saperated with a '|'.
6. `release_date`'s data type needed to be casted.
7. We could append a column for the movie `profit` using the formula: $profit = revenue - budget$.
8. `vote_average` better be presented as a catecorical variable that groubs multible ratings values.
9. We might also catigorize `profit` column for better EDA

## Exploratory Data Analysis:
After finishing our dataset cleaning, we endded up with a total of 10840 records and 10 columns. The dataset now has no duplicates nor null values, and the data types are consistant with suitable categorical variable to address our questions.
We then perfomed some analytics and created some visualizations to answer our targeted questions.
### Q1: Do movies with high popularity achive high revenvue?
> More popular movies recieve way more revenue than the less popular movies.

### Q2: What are the most filmed genres in this whole dataset?
> - `Drama`, `Comedy` and `Action` are the most three filmed genres in total of 10839 movies in our dataset.
> - `Drama` genre alone is filmed 22.6% of the times on our dataset.

### Q3: Is there a correlation between a movie budget and its revenue?
> There is positive correlation between `budget` and `revenue`, indecating a relation between them with little outliers found. 

## Built with:
- JupyterLab
- Python3
- Pandas
- Numpy

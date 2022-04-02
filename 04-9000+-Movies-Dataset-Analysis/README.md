# **Medical Appointment No Show Dataset Analysis**

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
This data set contains information about +9000 movies extracted from TMDB API. 

## Columns Descreption:
1. `Release_Date`: Date when the movie was released.
2. `Title`: Name of the movie.
3. `Overview`: Brief summary of the movie.
4. `Popularity`: It is a very important metric computed by TMDB developers based on the number of views per day, votes per day, number of users marked it as "favorite" and "watchlist" for the data, release date and more other metrics.
5. `Vote_Count`: Total votes received from the viewers.
6. `Vote_Average`: Average rating based on vote count and the number of viewers out of 10.
7. `Original_Language`: Original language of the movies. Dubbed version is not considered to be original language.
8. `Genre`: Categories the movie it can be classified as.
9. `Poster_Url`: Url of the movie poster.

## EDA Questions:
- Q1: What is the most frequent `genre` in the dataset?
- Q2: What `genres` has highest `votes`?
- Q3: What movie got the highest `popularity`? what's its `genre`?
- Q4: Which year has the most filmmed movies?

## Data Wrangling:
Our data can be found on `mymoviedb.csv` file provided on this repository, downloaded from [Kaggle](https://www.kaggle.com/datasets/disham993/9000-movies-dataset). 

## Data Cleaning:
### Exploration Summery
- we have a dataframe consisting of 9827 rows and 9 columns.
- our dataset looks a bit tidy with no NaNs nor duplicated values.
- `Release_Date` column needs to be casted into date time and to extract only the year value.
- `Overview`, `Original_Languege` and `Poster-Url` wouldn't be so useful during analysis, so we'll drop them.
- there is noticable outliers in `Popularity` column
- `Vote_Average` bettter be categorised for proper analysis.
- `Genre` column has comma saperated values and white spaces that needs to be handled and casted into category. 

We endded up with a datafram of a total of 6 columns and 25551 rows to dig into during our analysis after comleting our cleaning. 

## Data Visualization
Using `Matplotlib` and `Seaborn`, we made several meaningful visuals and charts to help us gain informative insights regarding any correlation between attributes in our dataset, that'll be discussed in the next section.

## Conclusion
These are derived conclusions after completing our data visualisation phase.

### Q1: What is the most frequent `genre` in the dataset?
`Drama` genre is the most frequent genre in our dataset and has appeared more than 14% of the times among 19 other genres.

### Q2: What `genres` has highest `votes`?
we have 25.5% of our dataset with popular vote (6520 rows).
`Drama` again gets the highest popularity among fans by being having more than 18.5% of movies popularities.

### Q3: What movie got the highest `popularity`? what's its `genre`?
`Spider-Man: No Way Home` has the highest popularity rate in our dataset and it has genres of `Action`, `Adventure` and `Sience Fiction`. 

### Q4: Which year has the most filmmed movies?
year `2020` has the highest filmming rate in our dataset.

## Built with:		
- JupyterLab	
- Python3	   	
- Pandas		
- Numpy			
- Matplotlib	
- Seaborn		

{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a href=\"https://www.kaggle.com/code/xshaimaa/9000-movies-dataset-analysis\" target=\"_blank\"><img align=\"left\" alt=\"Kaggle\" title=\"Open in Kaggle\" src=\"https://kaggle.com/static/images/open-in-kaggle.svg\"></a>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "tags": []
   },
   "source": [
    "## Dataset Description\n",
    "This data set contains information about +9000 movies extracted from TMDB API. \n",
    "\n",
    "## Columns Descriptions\n",
    "1. `Release_Date`: Date when the movie was released.\n",
    "2. `Title`: Name of the movie.\n",
    "3. `Overview`: Brief summary of the movie.\n",
    "4. `Popularity`: It is a very important metric computed by TMDB developers based on the number of views per day, votes per day, number of users marked it as \"favorite\" and \"watchlist\" for the data, release date and more other metrics.\n",
    "5. `Vote_Count`: Total votes received from the viewers.\n",
    "6. `Vote_Average`: Average rating based on vote count and the number of viewers out of 10.\n",
    "7. `Original_Language`: Original language of the movies. Dubbed version is not considered to be original language.\n",
    "8. `Genre`: Categories the movie it can be classified as.\n",
    "9. `Poster_Url`: Url of the movie poster.\n",
    "\n",
    "## EDA Questions\n",
    "- Q1: What is the most frequent `genre` in the dataset?\n",
    "- Q2: What `genres` has highest `votes`?\n",
    "- Q3: What movie got the highest `popularity`? what's its `genre`?\n",
    "- Q4: Which year has the most filmmed movies?\n",
    "___"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Environment Set-up"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5"
   },
   "outputs": [],
   "source": [
    "# importing lib.\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# getting dataset file dir.\n",
    "import os\n",
    "for dirname, _, filenames in os.walk('/kaggle/input'):\n",
    "    for filename in filenames:\n",
    "        print(os.path.join(dirname, filename))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "___\n",
    "### Public Functions\n",
    "here, we'd put all of the public functions to be used in this notebook"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**For usability and functionality sake, we would categorize columns using a function.**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def catigorize_col (df, col, labels):\n",
    "    \"\"\"\n",
    "    catigorizes a certain column based on its quartiles\n",
    "   \n",
    "    Args:\n",
    "        (df)     df   - dataframe we are proccesing\n",
    "        (col)    str  - to be catigorized column's name \n",
    "        (labels) list - list of labels from min to max\n",
    "    \n",
    "    Returns:\n",
    "        (df)     df   - dataframe with the categorized col\n",
    "    \"\"\"\n",
    "    \n",
    "    # setting the edges to cut the column accordingly\n",
    "    edges = [df[col].describe()['min'],\n",
    "             df[col].describe()['25%'],\n",
    "             df[col].describe()['50%'],\n",
    "             df[col].describe()['75%'],\n",
    "             df[col].describe()['max']]\n",
    "    \n",
    "    df[col] = pd.cut(df[col], edges, labels = labels, duplicates='drop')\n",
    "    return df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "___\n",
    "## Data Wrangling\n",
    "here, we'd load our data from the CSV file, and dive deeper into it to check for any required cleaning steps."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Release_Date</th>\n",
       "      <th>Title</th>\n",
       "      <th>Overview</th>\n",
       "      <th>Popularity</th>\n",
       "      <th>Vote_Count</th>\n",
       "      <th>Vote_Average</th>\n",
       "      <th>Original_Language</th>\n",
       "      <th>Genre</th>\n",
       "      <th>Poster_Url</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2021-12-15</td>\n",
       "      <td>Spider-Man: No Way Home</td>\n",
       "      <td>Peter Parker is unmasked and no longer able to...</td>\n",
       "      <td>5083.954</td>\n",
       "      <td>8940</td>\n",
       "      <td>8.3</td>\n",
       "      <td>en</td>\n",
       "      <td>Action, Adventure, Science Fiction</td>\n",
       "      <td>https://image.tmdb.org/t/p/original/1g0dhYtq4i...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2022-03-01</td>\n",
       "      <td>The Batman</td>\n",
       "      <td>In his second year of fighting crime, Batman u...</td>\n",
       "      <td>3827.658</td>\n",
       "      <td>1151</td>\n",
       "      <td>8.1</td>\n",
       "      <td>en</td>\n",
       "      <td>Crime, Mystery, Thriller</td>\n",
       "      <td>https://image.tmdb.org/t/p/original/74xTEgt7R3...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2022-02-25</td>\n",
       "      <td>No Exit</td>\n",
       "      <td>Stranded at a rest stop in the mountains durin...</td>\n",
       "      <td>2618.087</td>\n",
       "      <td>122</td>\n",
       "      <td>6.3</td>\n",
       "      <td>en</td>\n",
       "      <td>Thriller</td>\n",
       "      <td>https://image.tmdb.org/t/p/original/vDHsLnOWKl...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2021-11-24</td>\n",
       "      <td>Encanto</td>\n",
       "      <td>The tale of an extraordinary family, the Madri...</td>\n",
       "      <td>2402.201</td>\n",
       "      <td>5076</td>\n",
       "      <td>7.7</td>\n",
       "      <td>en</td>\n",
       "      <td>Animation, Comedy, Family, Fantasy</td>\n",
       "      <td>https://image.tmdb.org/t/p/original/4j0PNHkMr5...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2021-12-22</td>\n",
       "      <td>The King's Man</td>\n",
       "      <td>As a collection of history's worst tyrants and...</td>\n",
       "      <td>1895.511</td>\n",
       "      <td>1793</td>\n",
       "      <td>7.0</td>\n",
       "      <td>en</td>\n",
       "      <td>Action, Adventure, Thriller, War</td>\n",
       "      <td>https://image.tmdb.org/t/p/original/aq4Pwv5Xeu...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Release_Date                    Title  \\\n",
       "0   2021-12-15  Spider-Man: No Way Home   \n",
       "1   2022-03-01               The Batman   \n",
       "2   2022-02-25                  No Exit   \n",
       "3   2021-11-24                  Encanto   \n",
       "4   2021-12-22           The King's Man   \n",
       "\n",
       "                                            Overview  Popularity  Vote_Count  \\\n",
       "0  Peter Parker is unmasked and no longer able to...    5083.954        8940   \n",
       "1  In his second year of fighting crime, Batman u...    3827.658        1151   \n",
       "2  Stranded at a rest stop in the mountains durin...    2618.087         122   \n",
       "3  The tale of an extraordinary family, the Madri...    2402.201        5076   \n",
       "4  As a collection of history's worst tyrants and...    1895.511        1793   \n",
       "\n",
       "   Vote_Average Original_Language                               Genre  \\\n",
       "0           8.3                en  Action, Adventure, Science Fiction   \n",
       "1           8.1                en            Crime, Mystery, Thriller   \n",
       "2           6.3                en                            Thriller   \n",
       "3           7.7                en  Animation, Comedy, Family, Fantasy   \n",
       "4           7.0                en    Action, Adventure, Thriller, War   \n",
       "\n",
       "                                          Poster_Url  \n",
       "0  https://image.tmdb.org/t/p/original/1g0dhYtq4i...  \n",
       "1  https://image.tmdb.org/t/p/original/74xTEgt7R3...  \n",
       "2  https://image.tmdb.org/t/p/original/vDHsLnOWKl...  \n",
       "3  https://image.tmdb.org/t/p/original/4j0PNHkMr5...  \n",
       "4  https://image.tmdb.org/t/p/original/aq4Pwv5Xeu...  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# loading data and viewing its first 5 rows\n",
    "df = pd.read_csv('mymoviedb.csv', lineterminator='\\n')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 9827 entries, 0 to 9826\n",
      "Data columns (total 9 columns):\n",
      " #   Column             Non-Null Count  Dtype  \n",
      "---  ------             --------------  -----  \n",
      " 0   Release_Date       9827 non-null   object \n",
      " 1   Title              9827 non-null   object \n",
      " 2   Overview           9827 non-null   object \n",
      " 3   Popularity         9827 non-null   float64\n",
      " 4   Vote_Count         9827 non-null   int64  \n",
      " 5   Vote_Average       9827 non-null   float64\n",
      " 6   Original_Language  9827 non-null   object \n",
      " 7   Genre              9827 non-null   object \n",
      " 8   Poster_Url         9827 non-null   object \n",
      "dtypes: float64(2), int64(1), object(6)\n",
      "memory usage: 691.1+ KB\n"
     ]
    }
   ],
   "source": [
    "# viewing dataset info\n",
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- looks like our dataset has no NaNs!\n",
    "- `Overview`, `Original_Languege` and `Poster-Url` wouldn't be so useful during analysis\n",
    "- `Release_Date` column needs to be casted into date time and to extract only the year value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    Action, Adventure, Science Fiction\n",
       "1              Crime, Mystery, Thriller\n",
       "2                              Thriller\n",
       "3    Animation, Comedy, Family, Fantasy\n",
       "4      Action, Adventure, Thriller, War\n",
       "Name: Genre, dtype: object"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# exploring genres column\n",
    "df['Genre'].head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- genres are saperated by commas followed by whitespaces."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# check for duplicated rows\n",
    "df.duplicated().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- our dataset has no duplicated rows either."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Popularity</th>\n",
       "      <th>Vote_Count</th>\n",
       "      <th>Vote_Average</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>9827.000000</td>\n",
       "      <td>9827.000000</td>\n",
       "      <td>9827.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>40.326088</td>\n",
       "      <td>1392.805536</td>\n",
       "      <td>6.439534</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>108.873998</td>\n",
       "      <td>2611.206907</td>\n",
       "      <td>1.129759</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>13.354000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>16.128500</td>\n",
       "      <td>146.000000</td>\n",
       "      <td>5.900000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>21.199000</td>\n",
       "      <td>444.000000</td>\n",
       "      <td>6.500000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>35.191500</td>\n",
       "      <td>1376.000000</td>\n",
       "      <td>7.100000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>5083.954000</td>\n",
       "      <td>31077.000000</td>\n",
       "      <td>10.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "        Popularity    Vote_Count  Vote_Average\n",
       "count  9827.000000   9827.000000   9827.000000\n",
       "mean     40.326088   1392.805536      6.439534\n",
       "std     108.873998   2611.206907      1.129759\n",
       "min      13.354000      0.000000      0.000000\n",
       "25%      16.128500    146.000000      5.900000\n",
       "50%      21.199000    444.000000      6.500000\n",
       "75%      35.191500   1376.000000      7.100000\n",
       "max    5083.954000  31077.000000     10.000000"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# exploring summary statistics\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Exploration Summarey\n",
    "- we have a dataframe consisting of 9827 rows and 9 columns.\n",
    "- our dataset looks a bit tidy with no NaNs nor duplicated values.\n",
    "- `Release_Date` column needs to be casted into date time and to extract only the year value.\n",
    "- `Overview`, `Original_Languege` and `Poster-Url` wouldn't be so useful during analysis, so we'll drop them.\n",
    "- there is noticable outliers in `Popularity` column\n",
    "- `Vote_Average` bettter be categorised for proper analysis.\n",
    "- `Genre` column has comma saperated values and white spaces that needs to be handled and casted into category. \n",
    "___"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data Cleaning"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Casting `Release_Date` column and extracing year values**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "datetime64[ns]\n"
     ]
    }
   ],
   "source": [
    "# casting column a\n",
    "df['Release_Date'] = pd.to_datetime(df['Release_Date'])\n",
    "\n",
    "# confirming changes\n",
    "print(df['Release_Date'].dtypes)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dtype('int64')"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['Release_Date'] = df['Release_Date'].dt.year\n",
    "df['Release_Date'].dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Release_Date</th>\n",
       "      <th>Title</th>\n",
       "      <th>Overview</th>\n",
       "      <th>Popularity</th>\n",
       "      <th>Vote_Count</th>\n",
       "      <th>Vote_Average</th>\n",
       "      <th>Original_Language</th>\n",
       "      <th>Genre</th>\n",
       "      <th>Poster_Url</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2021</td>\n",
       "      <td>Spider-Man: No Way Home</td>\n",
       "      <td>Peter Parker is unmasked and no longer able to...</td>\n",
       "      <td>5083.954</td>\n",
       "      <td>8940</td>\n",
       "      <td>8.3</td>\n",
       "      <td>en</td>\n",
       "      <td>Action, Adventure, Science Fiction</td>\n",
       "      <td>https://image.tmdb.org/t/p/original/1g0dhYtq4i...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2022</td>\n",
       "      <td>The Batman</td>\n",
       "      <td>In his second year of fighting crime, Batman u...</td>\n",
       "      <td>3827.658</td>\n",
       "      <td>1151</td>\n",
       "      <td>8.1</td>\n",
       "      <td>en</td>\n",
       "      <td>Crime, Mystery, Thriller</td>\n",
       "      <td>https://image.tmdb.org/t/p/original/74xTEgt7R3...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2022</td>\n",
       "      <td>No Exit</td>\n",
       "      <td>Stranded at a rest stop in the mountains durin...</td>\n",
       "      <td>2618.087</td>\n",
       "      <td>122</td>\n",
       "      <td>6.3</td>\n",
       "      <td>en</td>\n",
       "      <td>Thriller</td>\n",
       "      <td>https://image.tmdb.org/t/p/original/vDHsLnOWKl...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2021</td>\n",
       "      <td>Encanto</td>\n",
       "      <td>The tale of an extraordinary family, the Madri...</td>\n",
       "      <td>2402.201</td>\n",
       "      <td>5076</td>\n",
       "      <td>7.7</td>\n",
       "      <td>en</td>\n",
       "      <td>Animation, Comedy, Family, Fantasy</td>\n",
       "      <td>https://image.tmdb.org/t/p/original/4j0PNHkMr5...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2021</td>\n",
       "      <td>The King's Man</td>\n",
       "      <td>As a collection of history's worst tyrants and...</td>\n",
       "      <td>1895.511</td>\n",
       "      <td>1793</td>\n",
       "      <td>7.0</td>\n",
       "      <td>en</td>\n",
       "      <td>Action, Adventure, Thriller, War</td>\n",
       "      <td>https://image.tmdb.org/t/p/original/aq4Pwv5Xeu...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Release_Date                    Title  \\\n",
       "0          2021  Spider-Man: No Way Home   \n",
       "1          2022               The Batman   \n",
       "2          2022                  No Exit   \n",
       "3          2021                  Encanto   \n",
       "4          2021           The King's Man   \n",
       "\n",
       "                                            Overview  Popularity  Vote_Count  \\\n",
       "0  Peter Parker is unmasked and no longer able to...    5083.954        8940   \n",
       "1  In his second year of fighting crime, Batman u...    3827.658        1151   \n",
       "2  Stranded at a rest stop in the mountains durin...    2618.087         122   \n",
       "3  The tale of an extraordinary family, the Madri...    2402.201        5076   \n",
       "4  As a collection of history's worst tyrants and...    1895.511        1793   \n",
       "\n",
       "   Vote_Average Original_Language                               Genre  \\\n",
       "0           8.3                en  Action, Adventure, Science Fiction   \n",
       "1           8.1                en            Crime, Mystery, Thriller   \n",
       "2           6.3                en                            Thriller   \n",
       "3           7.7                en  Animation, Comedy, Family, Fantasy   \n",
       "4           7.0                en    Action, Adventure, Thriller, War   \n",
       "\n",
       "                                          Poster_Url  \n",
       "0  https://image.tmdb.org/t/p/original/1g0dhYtq4i...  \n",
       "1  https://image.tmdb.org/t/p/original/74xTEgt7R3...  \n",
       "2  https://image.tmdb.org/t/p/original/vDHsLnOWKl...  \n",
       "3  https://image.tmdb.org/t/p/original/4j0PNHkMr5...  \n",
       "4  https://image.tmdb.org/t/p/original/aq4Pwv5Xeu...  "
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "___\n",
    "**Dropping `Overview`, `Original_Languege` and `Poster-Url`**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['Release_Date', 'Title', 'Popularity', 'Vote_Count', 'Vote_Average',\n",
       "       'Genre'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# making list of column to be dropped\n",
    "cols = ['Overview', 'Original_Language', 'Poster_Url']\n",
    "\n",
    "# dropping columns and confirming changes\n",
    "df.drop(cols, axis = 1, inplace = True)\n",
    "df.columns"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "___\n",
    "**categorizing `Vote_Average` column**\n",
    "\n",
    "We would cut the `Vote_Average` values and make 4 categories: `popular` `average` `below_avg` `not_popular` to describe it more using `catigorize_col()` function provided above."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['popular', 'below_avg', 'average', 'not_popular', NaN]\n",
       "Categories (4, object): ['not_popular' < 'below_avg' < 'average' < 'popular']"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# define labels for edges\n",
    "labels = ['not_popular', 'below_avg', 'average', 'popular']\n",
    "\n",
    "# categorize column based on labels and edges\n",
    "catigorize_col(df, 'Vote_Average', labels)\n",
    "\n",
    "# confirming changes\n",
    "df['Vote_Average'].unique()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "not_popular    2467\n",
       "popular        2450\n",
       "average        2412\n",
       "below_avg      2398\n",
       "Name: Vote_Average, dtype: int64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# exploring column\n",
    "df['Vote_Average'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Release_Date    0\n",
       "Title           0\n",
       "Popularity      0\n",
       "Vote_Count      0\n",
       "Vote_Average    0\n",
       "Genre           0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# dropping NaNs\n",
    "df.dropna(inplace = True)\n",
    "\n",
    "# confirming\n",
    "df.isna().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "___\n",
    "**Handling `Genre` column's comma saperated values**\n",
    "\n",
    "### TODO\n",
    "for this challenging column, we choose an approach that consists of stacking genres into a dataframe, and then merging it to our original dataframe. we'd explain further in the next cells."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "# creating a new dataframe that holds all genres for each movie\n",
    "#genres_df = df['Genre'].str.split(\", \", expand=True)\n",
    "\n",
    "# viewing its head\n",
    "#genres_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now that we have our dataframe of genres done, we'd move next into making a stack out of it, so that every movie would be represented by a stack of genres."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "# stacking genres dataframe \n",
    "#genres_df = genres_df.stack()\n",
    "\n",
    "# configuring it as pandas dataframe\n",
    "#genres_df = pd.DataFrame(genres_df)\n",
    "\n",
    "# viewing its first 10 rows\n",
    "#genres_df.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Renaming the genres column and confirming value count\n",
    "#genres_df.rename(columns={0:'genres_stack'}, inplace=True)\n",
    "#genres_df.genres_stack.value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now we have successfully created a new dataframe containing a stack of all movies' genres, we'd move into merging it with the original datarame\n",
    "___"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### we'd split genres into a list and then explode our dataframe to have only one genre per row for ezch movie"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Release_Date</th>\n",
       "      <th>Title</th>\n",
       "      <th>Popularity</th>\n",
       "      <th>Vote_Count</th>\n",
       "      <th>Vote_Average</th>\n",
       "      <th>Genre</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2021</td>\n",
       "      <td>Spider-Man: No Way Home</td>\n",
       "      <td>5083.954</td>\n",
       "      <td>8940</td>\n",
       "      <td>popular</td>\n",
       "      <td>Action</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2021</td>\n",
       "      <td>Spider-Man: No Way Home</td>\n",
       "      <td>5083.954</td>\n",
       "      <td>8940</td>\n",
       "      <td>popular</td>\n",
       "      <td>Adventure</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2021</td>\n",
       "      <td>Spider-Man: No Way Home</td>\n",
       "      <td>5083.954</td>\n",
       "      <td>8940</td>\n",
       "      <td>popular</td>\n",
       "      <td>Science Fiction</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2022</td>\n",
       "      <td>The Batman</td>\n",
       "      <td>3827.658</td>\n",
       "      <td>1151</td>\n",
       "      <td>popular</td>\n",
       "      <td>Crime</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2022</td>\n",
       "      <td>The Batman</td>\n",
       "      <td>3827.658</td>\n",
       "      <td>1151</td>\n",
       "      <td>popular</td>\n",
       "      <td>Mystery</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Release_Date                    Title  Popularity  Vote_Count Vote_Average  \\\n",
       "0          2021  Spider-Man: No Way Home    5083.954        8940      popular   \n",
       "1          2021  Spider-Man: No Way Home    5083.954        8940      popular   \n",
       "2          2021  Spider-Man: No Way Home    5083.954        8940      popular   \n",
       "3          2022               The Batman    3827.658        1151      popular   \n",
       "4          2022               The Batman    3827.658        1151      popular   \n",
       "\n",
       "             Genre  \n",
       "0           Action  \n",
       "1        Adventure  \n",
       "2  Science Fiction  \n",
       "3            Crime  \n",
       "4          Mystery  "
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# split the strings into lists\n",
    "df['Genre'] = df['Genre'].str.split(', ')\n",
    "\n",
    "# explode the lists\n",
    "df = df.explode('Genre').reset_index(drop=True)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "CategoricalDtype(categories=['Action', 'Adventure', 'Animation', 'Comedy', 'Crime',\n",
       "                  'Documentary', 'Drama', 'Family', 'Fantasy', 'History',\n",
       "                  'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction',\n",
       "                  'TV Movie', 'Thriller', 'War', 'Western'],\n",
       ", ordered=False)"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# casting column into category\n",
    "df['Genre'] = df['Genre'].astype('category')\n",
    "\n",
    "# confirming changes\n",
    "df['Genre'].dtypes"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "___"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 25552 entries, 0 to 25551\n",
      "Data columns (total 6 columns):\n",
      " #   Column        Non-Null Count  Dtype   \n",
      "---  ------        --------------  -----   \n",
      " 0   Release_Date  25552 non-null  int64   \n",
      " 1   Title         25552 non-null  object  \n",
      " 2   Popularity    25552 non-null  float64 \n",
      " 3   Vote_Count    25552 non-null  int64   \n",
      " 4   Vote_Average  25552 non-null  category\n",
      " 5   Genre         25552 non-null  category\n",
      "dtypes: category(2), float64(1), int64(2), object(1)\n",
      "memory usage: 849.4+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Release_Date     100\n",
       "Title           9415\n",
       "Popularity      8088\n",
       "Vote_Count      3265\n",
       "Vote_Average       4\n",
       "Genre             19\n",
       "dtype: int64"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.nunique()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now that our dataset is clean and tidy, we are left with  a total of 6 columns and 25551 rows to dig into during our analysis"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "___"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data Visualization\n",
    "here, we'd use `Matplotlib` and `seaborn` for making some informative visuals to gain insights abut our data."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "# setting up seaborn configurations\n",
    "sns.set_style('whitegrid') "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Q1: What is the most frequent `genre` in the dataset?\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count     25552\n",
       "unique       19\n",
       "top       Drama\n",
       "freq       3715\n",
       "Name: Genre, dtype: object"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# showing stats. on genre column\n",
    "df['Genre'].describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAWAAAAFuCAYAAAC/a8I8AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAABCrElEQVR4nO3de1zP9///8du7k6JylmMoJoc5m0PmvEaTOaUcojGb7YvJEClzXsjMMprTWEPU+MxmdqCRY8PHaUTkVIoip6Le1fv1+8PP+6MpC9Wrej+ul8sul3fv9+v1ej6fr9499vR8v1/3l0ZRFAUhhBCFzkjtDgghhKGSAiyEECqRAiyEECqRAiyEECqRAiyEECqRAiyEECqRAizEcwQGBjJ79mxV+3D69Gm6desGwKZNm1i5cuVztw8NDWXDhg05vvb0/t26deP06dMv1JfY2FjGjRsHwM2bN3F3d3+h/UV2Jmp3QAiRd4MHD/7XbY4dO0b9+vVfev/niY+P5/LlywDY2NgQEhLySsczdFKARaFYuXIlYWFhlClThtatW7N7927Cw8PRarUEBARw5MgRsrKyaNSoEb6+vlhaWtKtWzf69evHoUOHSEhIoFevXkyZMoXIyEjmzZtH6dKlefjwIWFhYezfv58VK1aQkZGBubk53t7etGjR4pl+/Pnnn3z55ZfodDpKly7NrFmzcHBwYNeuXSxbtoysrCwsLS2ZNm0aTZs2zbZvt27dWLp0Ka+//nq2n8uXL8+IESNo164dJ06cIDMzkylTprB582YuXbpEkyZN+OKLL4iPj8fT05POnTtz8uRJ7t27h5eXF87Ozs/0c+PGjaxfvx5LS0tee+01/fOBgYHcuXOHGTNmsHHjRkJCQjA1NaVUqVLMnj2by5cvEx4ezoEDBzA3Nyc5OZkTJ06QmJhIgwYNqF27tn7/J+2cO3cOrVbLe++9x8CBA4mMjGTOnDn8/PPPAPqff/zxR3x9fbl58yajRo1i1qxZuLi4cPz4cTIyMvD39+fQoUMYGxvTtGlTpk2b9tzfowAUIQpYRESE8vbbbyv37t1TdDqdMm3aNKVr166KoihKYGCg4u/vr+h0OkVRFGXx4sXKZ599piiKonTt2lXx9/dXFEVRbty4obz++uvKtWvXlMOHDysODg5KXFycoiiKcvnyZaV3795KcnKyoiiKEh0drTg6OiqpqanZ+pGUlKS0atVKOXv2rKIoivLbb78po0aNUi5evKh06NBBuXbtmqIoinLw4EHF0dFRefDggfLVV18ps2bN0vfn1KlT+uM9+Tk2NlZ57bXXlF27dimKoigzZsxQunbtqjx48EBJS0tTHB0dlWPHjum3Cw8PVxRFUX799VelS5cuz5yvs2fPKu3bt1cSExMVRVEUPz8//fl60p/MzEylcePGys2bNxVFUZRt27YpISEhiqIoire3t7J69Wr99m+//baSkZGRbf8n/X9yrm/cuKG0a9dOiY6OVg4fPqy88847+v48/fPTj2NjY5XmzZsriqIoS5cuVcaOHatotVolKytLmTp1quLn5/fc36NQFJkBiwK3d+9eevbsibW1NQBDhw7l8OHDAOzZs4cHDx5w8OBBADIyMqhYsaJ+3+7duwOP/7lbsWJF7t27B0C1atWoUaMGAAcOHCAxMRFPT0/9fhqNhmvXruHg4KB/7r///S/169enYcOGADg5OeHk5MSGDRto164dtWrVAqB9+/ZUqFCBv//+O89jNDU11a/T2tra0qJFCywtLQGoUqUK9+7do0qVKpiamtK5c2cAGjVqxN27d5851qFDh3B0dKRy5coAuLm5sX///mzbGBsb07NnT9zd3enSpQuOjo64uLjk2LfmzZtjYpLzn/qTNVwbGxs6duzIoUOHaNCgQZ7H/URERAReXl6YmpoC4OHhwf/93//pX8/p9/jkfBsyKcCiwJmYmKA8FTlibGysf6zT6fDx8dEXpdTUVNLT0/WvlypVSv9Yo9Hoj1O6dOlsx2jfvj1ffvml/rmEhASqVKmSrR/GxsZoNBr9z4qicP78+Wx9e/q1zMzMHJ9/QqvV6h+bmppmO/aTQvRPpqamGBkZ6ceTk6fH+aTfOQkICCA6OpqDBw+yatUqwsLCWLFixTPbPX2u/ulJX+Dx2ExMTJ5pPyMjI9f9n9DpdM/8/PR+uf0eDZ18C0IUuM6dO/P777/z4MEDAMLCwvSvdezYkQ0bNqDVatHpdPj5+fHFF1+80PHbtWvHgQMHiImJAR7PuPv06ZOtkAM0a9aMmJgYLly4AMDu3buZPHmyfv/Y2FgA/Vpls2bNsu3/9Kz4xIkTJCUlvVA/86pDhw4cOHCAGzduALBt27ZntklOTqZz586UK1cOT09PJkyYwPnz54HHBTun/3nk5Mmx4+PjOXjwoH72Hx8fz+3bt1EUhV27dum3NzY2zrEgv/nmm4SEhJCRkYFOp2PDhg04Ojq+8NgNjcyARYFr3749gwYNws3NDXNzc+rXr4+FhQUAH3/8MQsWLKBfv35kZWXRsGFDpk6d+kLHr1+/PrNnz2bixIn6WdyKFSuemflVqlSJgIAAvL299R+2LVmyhHr16vHZZ58xduxYsrKyMDc3JygoCCsrq2z7T5o0iZkzZ7J582YaN25M48aNX+3E5KJBgwZMnjyZESNGUKZMmWc+DITH/zP46KOP8PT0xNzcHGNjY+bOnQtAp06dmDNnTp7aSk9Pp1+/fmRkZODr60vdunWBx0sTAwYMoHLlynTp0kW/ff369TE2NmbgwIEsWbJE//xHH33EggUL6Nu3L5mZmTRt2hQ/P79XOAuGQaPIvwVEATt9+jTHjx9n+PDhAHz77becPHky25KBEIZICrAocCkpKfj4+HDp0iU0Gg3VqlVjzpw52NjYqN01IVQlBVgIIVQiH8IJIYRKpAALIYRKpACr6NSpU6q2f+XKFWlfZWr3QdpXt30pwCrKyspStf1Hjx5J+ypTuw/Svrrty4dwKjpz5iyNGzdSuxtCiBekzdRhZvLq81e5EENFRkYaBi++onY3hBAvaNOndfLlOLIEIYQQKpECLIQQKjGoJYjIyEgmTJhAvXr19GlXw4cPzzEQWwghCppBFWB4nJz1JEQkNTUVDw8P6tatq8+IFUKIwmJwBfhpZcqUwc3NjdmzZ5OZmYmpqSmDBg3C3NycDRs2kJmZiUajYdmyZVy4cIGVK1diamrKjRs3cHd35/Dhw5w7d47hw4czZMgQfv3112f2q1ChgtrDFEIUUQZdgAEqVqzInTt3MDMzIzQ0FICgoCBWrlyJhYUFM2bMYP/+/djY2HDjxg3+85//cObMGT755BP++OMPbt68ydixYxkyZAhXrlx5Zr8+ffqoPEIhRFFl8AU4Pj6ePn366MOs4XFR9vb2pkyZMly6dInmzZsDj7NQTU1NsbKywtbWFjMzM8qWLasP/s5tPyGEyIlBF+CUlBRCQ0MZOnSo/tYsDx484KuvvmLPnj0AvPfee/rbp+R2C5l/208IIXJicAX48OHDeHh4YGRkRFZWFuPGjaNs2bJERkYCYGlpScuWLXFzc8PExARra2sSExOpWbPmc4+b235CCJEbuRRZRVFRUcz+xULtbgghXpBcCSeEEMWcwS1BFCU6nZJv/ycVQhSe/ArjkRmwirTa9H/fqABFRUVJ+ypTuw/S/su1nx/FF6QACyGEauRDOBVJHrAwdA/TMihtbqpa+1FRUarGEMgasIokD1gYOkP/DESWIIQQQiVSgIUQQiUlbgniwoULLFq0iEePHvHw4UM6d+7MuHHjnnsZ8avYtGkTt27dYty4cQVyfCFEyVWiZsD3799n4sSJ+Pj4EBwczJYtW4iOjiYkJETtrgkhxDNK1Ax49+7dtG3bljp16gBgbGzMggULMDU1xd/fn2PHjgHQu3dvRowYwdSpUzExMSE+Ph6tVouzszN//vknCQkJLF++HFtbWxYvXszRo0fR6XR4enrSq1cvjh49yvz587G2tsbY2JjmzZuzefNmrly5gre3N1lZWfTt25ewsDBKlSql4hkRQhRlJWoGnJiYSK1atbI9V6ZMGQ4cOEBcXBxbtmxh48aN/Pzzz/r4yRo1arB27Vrs7OyIi4tj1apVODk5ER4ezt69e4mLi2PTpk189913BAUFcf/+fWbNmsXixYtZt26dPqTnnXfeYffu3WRlZbFv3z7atm0rxVcI8VwlagZcvXp1zp49m+252NhYzpw5Q+vWrdFoNJiamtKsWTNiYmIAaNTo8fdwra2tsbOz0z/WarVER0dz5swZPDw8AMjMzOT69evcunWLunXrAtCyZUuuXbuGpaUlbdq0Yf/+/WzdupWPP/64sIYthCimStQMuGvXruzbt49r164BkJGRgb+/P9bW1vrlh4yMDI4fP07t2rWB52f82tnZ0bZtW4KDg1m/fj29evWiVq1a2NjY6Av46dOn9dsPGjSI0NBQbt++jYODQ0ENUwhRQpSoGbClpSX+/v74+vqiKAqpqal07doVDw8PEhIScHNzIyMjg549e9K4ceN/PV63bt3466+/GDJkCA8fPqRHjx5YWloye/ZspkyZgqWlJWXKlKFs2bIANGvWjKtXrzJ06NCCHqoQogSQS5HzkU6nY/DgwaxZswZLS8t/3V7ygIWhU/tKOLUvRS5RSxBqio2NpV+/fjg7O+ep+AohhMyAVSRhPMLQGXoYj8yAVSR5wIbdflHog9rtX718UdX21SYFWAghVCIFWAghVCIFWEVmZupeKafm2pe0XzT6kJf2tZm6QuiJYSpR3wMubiSQXRQHan9VrCSTGbAQQqjEoAvwqlWr6NixI+npuX8b4fz58xw5cgQALy8vtFptYXVPCFHCGXQB3r59O87OzuzYsSPXbX7//XcuXnz8VZklS5ZgZmZWWN0TQpRwBrsGHBkZia2tLe7u7kyePJn+/ftz8uRJ5s+fj06nw8bGBj8/P7Zt24apqSmNGzdmwoQJ7Ny5k6SkJHx8fMjKykKj0eDr64uDgwNOTk60bNmSy5cvU7FiRQIDAzE2NlZ7qEKIIspgC3BoaCiurq7Y2dlhZmbGyZMnmTFjBl988QX29vaEhoZy69Yt+vXrR6VKlWjatKl+34ULFzJ8+HB69OhBVFQUPj4+bN26ldjYWNavX0+1atVwd3fn9OnTNG/eXL1BCiGKNIMswPfu3SMiIoLk5GSCg4NJSUnh+++/59atW9jb2wPg6uoKQHh4+DP7x8TE0KZNG+Dx13hu3LgBQPny5alWrRoA1apVe+7ashBCGGQB3r59OwMGDMDb2xuAR48e0b17d8zNzbly5Qp16tRh5cqV1K1bF41Gg06X/XuQ9vb2HD16lO7duxMVFUWlSpWA52cLCyHEPxlkAQ4NDWXhwoX6ny0sLHBycqJSpUr4+PhgZGRE5cqV8fT0xNTUlIULF+pnxgBTpkzBz8+PtWvXkpmZybx589QYhhCimJM0NBVJHrAoDgryQgy108jUbt+gv4YmhBBqkgIshBAqMcg14KJCp1PkOntR5GkzdZiZyFytIMhZVZEEsht2+0WhD3lpX4pvwZEzK4QQKpECrCLJAzbs9tXsg2T8Fg2yBqwiyQMWapHPHooGmQELIYRKpAALIYRKDGYJwt/fnzNnzpCUlERaWhq1atXiwoULtG/fniVLluS6X0REBAkJCTg6OjJx4kS2bNlCt27d2LlzJ6VKqbuGK4Qo3gymAE+dOhWArVu3cunSJSZNmkRkZCQhISHP3a9Tp04AxMXFFXgfhRCGxWAKcG6uXr3K+++/T3JyMl27dmXcuHF4eHhQoUIF7t27xzvvvMPVq1dxd3d/Zt+EhAT8/PxIT0+nVKlSzJkzh6ysLD766CPKlStHp06dGD16tAqjEkIUBwZfgNPT01m+fDlZWVl06dKFcePGAdC7d2/eeusttm7dmuu+CxYswMPDg86dO3Po0CECAgLw8vIiKSmJH374QW5fJIR4LoMvwPXr19cXShOT/52OunXr/uu+0dHRfPPNN6xevRpFUfT716xZU4qvEOJfGXwBzi1EPS/h6nZ2dowcOZKWLVsSExOjv3uykZF8uUQI8e8MvgC/Cm9vb2bOnEl6ejppaWlMnz5d7S4JIYoRCWRXkQSyC7U8uRJO7UByQ29f/q0shBAqkSUIFUkesFCLZPwWDfIbUJHkARt2+2r2QYpv0SC/BSGEUIkUYBVJHnDJbF+ydkVeyRqwiiQPuGSSdX2RVzIDFkIIlUgBFkIIlRTLArxq1So6duxIevqz3yLYtGkTgYGB+dLOH3/8wc2bN/PlWEII8U/FsgBv374dZ2dnduzYUaDtfPfdd6SkpBRoG0IIw1XsCnBkZCS2tra4u7uzYcMGAI4ePUr//v3x9PRk165dwOPiuWzZMgC0Wi09e/ZEq9USHByMm5sb7u7ufPfdd8DjsPYZM2YwatQoXFxcOHPmDHv27CEqKgpvb28uX77MoEGD9H0YNGgQcXFxBAYGMnLkSNzd3YmJicnx2EIIkZtiV4BDQ0NxdXXFzs4OMzMzTp48yaxZs1i8eDHr1q2jZs2aALz77rvs3LkTRVHYvXs3Xbt25dq1a/zyyy9s3LiRDRs2sGvXLi5dugRA9erVWbNmDR4eHmzevJkuXbrQsGFDFixYgKmpaa79sbOzIyQkBEVRcj22EELkpFh9De3evXtERESQnJxMcHAwKSkpfP/999y6dUuf39uyZUuuXbtG2bJladiwIceOHWPbtm14e3tz/vx54uPj8fT01B/v6tWrwP++E1q1alX++9//PrcfT+cXPWk3Ojo6x2Pb2dnl5ykQQpQgxaoAb9++nQEDBuDt7Q3Ao0eP6N69OxYWFsTExGBvb8/p06cpW7Ys8HipYP369aSlpWFvb09GRgb16tVj9erVaDQa1q1bR4MGDfjtt99yzP/VaDQoikKpUqW4ffs2WVlZpKamZrs/3JPsXzs7uxyPLYQQuSlWBTg0NJSFCxfqf7awsMDJyYlKlSoxZcoULC0tKVOmjL4Av/HGG/j5+fHRRx8B4ODgQPv27Rk8eDBarZamTZtiY2OTa3stWrRgypQprF27FkdHRwYOHEitWrWoXbv2M9u+6LGFEELygFUkecAl04tcCad2Hq20L3nAQghhkIrVEkRJI3nAJZNk7Yq8kneJiiQPuGS2L8VX5JW8U4QQQiVSgFUkecDqtC95vaKokDVgFUkesDpk3V0UFTIDFkIIlUgBFkIIlRTbJYjIyEgmTJhAvXr1AEhNTaVmzZoEBARgZmamcu+EEOLfFesZcLt27QgODiY4OJitW7diampKeHi42t0SQog8KbYz4H/SarUkJiZStmxZ/P39OXbsGAC9e/dmxIgRTJ06FRMTE+Lj49FqtTg7O/Pnn3+SkJDA8uXLqVGjBjNmzODGjRskJibSrVs3vLy8mDp1KmZmZly/fp3ExET8/f1p3LgxoaGhbNq0CZ1OR7du3Rg/fjw7d+5k3bp1GBkZ0apVKyZNmqTyWRFCFGXFegZ8+PBhPDw8cHZ2pn///rz11lukpaURFxfHli1b2LhxIz///DPnz58HoEaNGqxduxY7Ozvi4uJYtWoVTk5OhIeHk5CQQPPmzVmzZg1hYWGEhITo2/lnVvDt27dZtWoVGzduZNu2bWi1WuLj4wkMDGTdunVs2rSJmzdvcuDAAbVOjRCiGCjWM+B27dqxZMkS7ty5w8iRI6lZsyYxMTG0bt0ajUaDqakpzZo1IyYmBoBGjRoBYG1trc/ptba2RqvVUq5cOU6fPs3hw4extLREq9Xq2/lnVnBsbCz169fH3NwcgEmTJnHq1CmSk5P54IMPgMdr0teuXcPR0bHQzocQongp1jPgJ8qXL8+iRYvw9fWlUqVK+uWHjIwMjh8/ro+PzCnz94mtW7diZWXF4sWLGTlyJGlpafrg9X/uZ2try6VLl/RFevz48VSsWJFq1aqxdu1agoODGTZsGM2bNy+A0QohSopiPQN+Wr169fDw8CA8PJyaNWvi5uZGRkYGPXv2pHHjxv+6f/v27fn00085ceIEZmZm1K5dm8TExBy3rVChAqNHj2bYsGFoNBq6du1KjRo18PT0xMPDg6ysLGrUqEGvXr3ye5hCiBJE8oBVJHnA6nhyJZzaWbBFoQ/SvuQBCyGEQSoxSxDFkeQBq0PyekVRIe9CFUkesDrtS/EVRYW8E4UQQiVSgIUQQiVSgFUkgewSyC4Mm3wIpyIJZFeHfPApigqZAQshhEqkAAshhEoMrgBHRkbi5eWV7bmAgAC2bt2qUo+EEIbK4AqwEEIUFfIh3FNyC3K/e/cud+/eZdSoUaxcuRJTU1MGDRpE5cqV+fLLLylVqhTlypVj/vz5REVFERAQoN+mb9++6g5KCFFkGWQBfhLk/kRsbCzvv/++Psg9MzOTIUOG0K5dO+Bx7rCnpyeRkZGkp6cTGhqKoih0796dTZs2YWNjw/r161mxYgVdunTRbyOEEM9jkEsQT99LLjg4mN69e5OWlpZrkHvdunX1+z55fOfOHSwtLbGxsQGgTZs2XLhw4ZnthRAiNwZZgHNibm6epyB3I6PHp6x8+fKkpKToM4P/+usv6tSpk20bIYR4HoNcgshJ6dKlXyjIXaPRMHfuXMaNG4dGo6Fs2bJ8/vnn+lmwEEL8GwlkV5EEsqtDAtml/aLSvvxbWQghVCJLECqSQHZ1SCC7KCrkXagiCWSXQHZh2OSdKIQQKpECrCLJAy7c9iUHWBQ1sgasIskDLlyy3i6KGpkBCyGESqQACyGESoptAV61ahUdO3YkPT33bxIkJSUxc+bMV2rnyJEjnDt3DoCxY8e+0rGEEOJpxbYAb9++HWdnZ3bs2JHrNpUrV37lAvzDDz/o8x6WLVv2SscSQoinFcsP4SIjI7G1tcXd3Z3JkyfTv39/PDw8cHBw4MKFC6SkpLB06VIURWHixIls2bIFFxcXWrduzfnz57Gzs6NixYocPXoUMzMzVq5cye3bt5k5cybp6ekkJSUxYcIEqlatyr59+zhz5gz16tXD1dWVAwcOcPbsWebMmYOxsTGlSpVizpw56HQ6Pv30U6pWrUpsbCyvv/46s2bNUvtUCSGKsGI5Aw4NDcXV1RU7OzvMzMw4efIkAE2bNmXdunU4Ojo+MzNOTU2ld+/ebNy4kaNHj9KyZUs2bNhARkYGFy9e5NKlS7z33nt8++23zJ49mw0bNtCkSRPefPNNJk+eTPXq1fXH8vX1ZcaMGXz//fcMHjwYf39/AK5cucK8efMIDQ0lIiKCpKSkwjspQohip9jNgO/du0dERATJyckEBweTkpLC999/D0CjRo0AqFq1Krdu3Xpm3yfpZtbW1tjb2+sfp6enU7lyZVasWEFYWBgajYbMzMxc+5CYmKj/DmubNm1YvHgxALa2tlhaWgKPlz+etz4thBDFrgBv376dAQMG4O3tDcCjR4/o3r075cuX/9d9n871/aelS5fi6upK586d+eGHH9i2bZt+n38GxlWpUoVz587h4ODAkSNH9DnAzzu+EEL8U7ErwKGhoSxcuFD/s4WFBU5OToSFhb3ScXv27MnChQtZuXIlVatW5c6dOwA0a9aMgIAAatasqd927ty5zJkzB0VRMDY2Zv78+a/UthDCMEkesIokD7hw/fNKOLWzYItCH6R9yQMWQgiDVOyWIEoSyQMuXJIDLIoaeTeqSPKAC7d9Kb6iqJF3pBBCqEQKsIokD7hw2pccYFFUyRqwiiQPuHDIOrsoqmQGLIQQKpECLIQQKjGoJYi4uDj69Omjz4QAaNu27Uvl/EZFRbF7927Gjh2Lo6MjBw4cyM+uCiEMgEEVYIB69eoRHBz8ysdp2LCh6h9iCSGKN4MrwP+UlZXFjBkzuHHjBomJiXTr1g0vLy+mTp2KiYkJ8fHxaLVanJ2d+fPPP0lISGD58uUkJCQQEhLCkiVLAHjw4AH9+vXjt99+w9jYmEWLFtG4cWOcnZ1VHqEQoqgyuDXgixcv4uHhof/vxIkTNG/enDVr1hAWFkZISIh+2xo1arB27Vrs7OyIi4tj1apVODk5ER4e/sxxraysaNWqFfv37ycrK4uIiAh69OhRmEMTQhQzBjcD/ucSREpKCj/++COHDx/G0tISrVarf+1JvrC1tTV2dnb6x09v8zRXV1eCg4PR6XR06NABMzOzAhyJEKK4M7gZ8D9t3boVKysrFi9ezMiRI0lLS9Pn/75ovm/r1q2JjY0lLCyMgQMHFkR3hRAliMHNgP+pffv2fPrpp5w4cQIzMzNq166tvwnny3BxceHXX3+lfv36+dhLIURJZFAFuGbNmmzZsiXbc/Xr12f79u3PbPvkPm8AkyZN0j/29PTUP27bti1Atq+gZWVl4erqml9dFkKUYHkuwFeuXOHq1as0aNAAGxsbuf1ODqZOnUpiYiJBQUFqd0UIUQzkqQB///33/PHHH9y7d4++ffty7do1ZsyYUdB9K3aenjXnheQBFw7JARZFVZ7elTt27ODbb7/FysoKT09P/W3gxauRPODCaV+Kryiq8vTOVBQFjUajX3aQr1cJIcSry9MSRO/evRk6dCjx8fGMHj1aLjAQQoh8kKcC3KFDB9q3b090dDR169bFwcGhoPtlECSQPf/al3VeURzlqQBPnz6dTZs2YW9vX9D9MSgSyJ5/5MNMURzlqQCXLl2a+fPnU7duXYyMHs8y3NzcCrRjQghR0uWpALdo0QKA27dvF2hn8kt+5P7+8ccfNG3aFBsbm4LoohBC5K0Ajx07ltu3b5Oeru7Xpl7Eq+b+fvfdd8ycOVMKsBCiwOSpAM+aNYu9e/dSpUoV/VfSno5tLA6el/trZmbG9evXSUxMxN/fn6SkJKKiovD29mbjxo0EBgby999/c/fuXRwcHPj88885duwYCxYswMTEBAsLC5YuXcpnn32Gi4sLXbp0ISYmhgULFrBy5Uq1hy6EKKLyVIBPnjzJrl279Ou/xcGT3N8nJkyYQPPmzXF1dSU9PZ1OnTrh5eUFQPXq1Zk9ezZbtmxh8+bNzJ49m4YNGzJz5ky0Wi3W1tZ8++236HQ63nnnHW7evMmuXbvo1asXI0aMIDw8nPv37+Pq6sqmTZvo0qWLJKIJIf5Vngpw7dq1SU9Px8LCoqD7k29eJPf3ydehqlatyn//+99sxylVqhTJyclMnDiR0qVL8/DhQzIyMhgzZgxBQUGMGDECGxsbmjZtStu2bZk7dy7JyckcOHCAiRMnFs5ghRDFUp6mtAkJCXTt2hU3Nzfc3Nxwd3cv6H7luxfN/dVoNCiKQkREBAkJCXzxxRdMnDhRv9/27dvp168fwcHB1K9fny1btqDRaOjTpw9z587F0dERU1PTwh6mEKIYydMMePHixQXdjwL3orm/LVq0YMqUKaxYsYLly5czdOhQNBoNtWrVIjExkaZNm+Lr64uFhQVGRkbMnj0bgP79+9OlSxd+/PHHwhqaEKKY0ihPpoHPcfPmTRYtWkRycjI9e/akQYMGNGvWrDD6V+zcvHmTKVOmsH79+n/dNioqitm/FJ9lnaLsZS7EiIqKUv1qQLX7IO2r236eliD8/PwYMGAAGRkZtG7dmnnz5hV0v4ql33//nffff5/x48er3RUhRDGQpwKclpZG+/bt0Wg02NnZUaqUuhkGRZWTkxM//fQTrVq1UrsrQohiIE9rwKVKlWLfvn3odDr9Gqp4dRLInn8kjEcUR3l6x86ZM4c1a9Zw5swZlixZwqxZswq6XwZBAtnzr30pvqI4eu679uLFiwwfPpyqVaty48YNXnvtNa5cucLZs2cLq39CCFFiPbcABwQEMHnyZAAqV67M5s2b+e67714pY0H8j+QBv3r72kxdPvRECHU8dw340aNHvP766wBYWVkBj6+Ky8zMLPieGQDJA351soYuirPnzoCfTj9bvny5/rGJSZ7vZi+EECIXzy3AVapU4dSpU9meO3XqFJUrVy7QTgkhhCF47lR28uTJfPzxx7Rr147atWsTGxvLoUOHCAoKeu5BV65cycGDB8nMzESj0eDt7U2TJk1y3HbevHm89957VK9e/eVHkUfdunWjWrVq+lS3smXLsmzZMsaOHcuyZcty3Of8+fPcv3+fNm3a4OXlxYIFC+RreEKIfPHcAlyrVi1CQ0MJDw8nLi6OJk2a8Mknn1C6dOlc97l48SLh4eFs2rQJjUajz9Xdvn17jttPnz791UbwgtauXfvMhSS5FV94fHVbpUqVaNOmDUuWLCno7gkhDMi/Luaam5vj7Oyc5wNaWVkRHx9PWFgYnTp1omHDhoSFhQGPc4Xnz5+PTqfDxsaGgIAARo8ezcyZM6lSpQrTp0/nzp07APj6+tKgQQOcnJxo2bIlly9fpmLFigQGBpKRkcG0adOIj48nIyMDPz8/mjRpwmeffcbVq1fR6XRMmDCBtm3b5qnPjo6OHDhw4Jn++fn5sW3bNkxNTWncuDETJkxg586dJCUl4ePjQ1ZWFhqNBl9fXxwcHHLsq7GxcZ7PnRDCsOT7p2k2NjasWLGC77//nq+//hpzc3O8vLx4++23mTFjBl988QX29vaEhoYSExOj3y8oKIh27doxZMgQrly5wrRp09i0aROxsbGsX7+eatWq4e7uzunTpzlx4gQ1atRgyZIlXLlyhT179hAVFUX58uWZP38+d+7cYdiwYezYseOZ/o0cOVK/BDFq1Ci6dOmif+2f/bt16xb9+vWjUqVKNG3aVL/dwoULGT58OD169CAqKgofHx+2bt2aY1+bN2+e36dYCFFC5HsBvnr1KpaWlnz++ecAnD59mtGjR9O2bVtu3bqlv7W9q6trtv2io6M5fPgwO3fuBODevXsAlC9fnmrVqgFQrVo10tPTuXTpEp06dQKgTp06eHp6MnPmTI4dO6b/0DAzM5Pk5GQqVKiQrZ2cliCeyKl/4eHhz2wXExNDmzZtgMffZb1x40aufRVCiNzkewE+f/48mzdvZsWKFZiZmVG3bl2sra0xNjamSpUqXLlyhTp16rBy5Urq1q2r38/Ozo4+ffrg4uLC7du3CQ0NBXIOS7e3t+f06dP06NGD2NhYvvzyS5o1a0bVqlUZM2YMaWlprFixgnLlyr1Q33Pqn0ajQafL/mV/e3t7jh49Svfu3YmKiqJSpUq59lUIIXKT7wXYycmJmJgYBg4cSOnSpVEUhSlTpmBlZcWsWbPw8fHByMiIypUr4+npyXfffQfAmDFjmD59Olu2bCElJeW5t5B3d3fHx8eHYcOGkZWVhY+PDw0aNMDX15dhw4aRkpLCkCFDXvgedjn1z9TUlIULF+pnxgBTpkzBz8+PtWvXkpmZKfGcQoiXkqdAdlEwJJD91b3KlXBqh3EXhT5I+8UgkF0IIUT+k2uKVSR5wK9OcoBFcSbvXBVJHvCrty/FVxRn8u4VQgiVSAFWkeQBv3z7kgMsSgJZA1aR5AG/PFk7FyWBzICFEEIlUoCFEEIlBl2AL1y4wAcffICHhwcDBgzgq6++4p/XpXh5eaHValXqoRCiJDPYNeD79+8zceJEAgMDqVOnDllZWXzyySeEhIQwePBg/XaSASyEKCgGW4B3795N27ZtqVOnDgDGxsYsWLCA48eP4+rqiqmpKYMGDeKrr75i586dfPbZZ5iYmBAfH49Wq8XZ2Zk///yThIQEli9fjq2tLYsXL+bo0aPodDo8PT3p1auXuoMUQhRpBrsEkZiYSK1atbI9V6ZMGUxNTUlPT2fjxo307ds32+s1atRg7dq12NnZERcXx6pVq3ByciI8PJy9e/cSFxfHpk2b+O677wgKCuL+/fuFOCIhRHFjsDPg6tWrc/bs2WzPxcbGcuTIkWwxmU9r1KgRANbW1tjZ2ekfa7VaoqOjOXPmDB4eHsDjPOLr169jbW1dgKMQQhRnBjsD7tq1K/v27ePatWsAZGRk4O/vT/ny5XONsXxe3q+dnR1t27YlODiY9evX06tXr2dm2EII8TSDnQFbWlri7++Pr68viqKQmppK165d9WHrL6pbt2789ddfDBkyhIcPH9KjRw8sLS0LoOdCiJJC8oBVJHnALy8/roRTOwu2KPRB2pc8YCGEMEgGuwRRFEge8MuTHGBREsg7WEWSB/zy7UvxFSWBvIuFEEIlUoBVVBLygCWXV4iXJ2vAKioJecCyhi3Ey5MZsBBCqEQKsBBCqKTEFuDIyEgaNGjAjh07sj3v4uLC1KlT83ycI0eOcO7cufzunhBClNwCDI/zGZ4uwOfPn+fRo0cvdIwffviBxMTE/O6aEEKU7A/hHBwcuHz5Mg8ePMDKyort27fj4uJCaGgo48eP56uvvgLA3d2dpUuX8uWXX3L16lXS0tIYPnw49erVY9++fZw5c4Z69epx8uRJ1q1bh5GREa1atWLSpEkEBgZy/PhxHj58SK9evbhx4wbe3t5kZWXRt29fwsLCKFVK3W87CCGKphI9AwZwcnLi999/R1EUTp06RYsWLXB0dCQ6Opp79+5x4cIFypcvT5kyZThy5AjLli1j9erVGBsb06RJE958800mT55M6dKlCQwMZN26dWzatImbN29y4MAB4PFMOyQkhAEDBrB7926ysrLYt28fbdu2leIrhMhViZ4Bw+M135kzZ1KrVi1at24NPI6V7NOnDz///DNxcXEMHDgQS0tLfHx88PPzIyUlhT59+mQ7zrVr10hOTuaDDz4AIDU1VR9l+SQ/2NLSkjZt2rB//362bt3Kxx9/XIgjFUIUNyV+BlyrVi0ePnxIcHBwtqI6YMAAfv31V44cOULnzp1JTEzkzJkzfP3116xcuZJFixaRmZmJRqNBURRq1qxJtWrVWLt2LcHBwQwbNozmzZsDZMsPHjRoEKGhody+fRsHB4fCHq4Qohgp8QUYwNnZmYSEhGx3urCxsaFMmTK0b98eExMTKleuTFJSEu7u7rz33nuMHDkSExMTmjVrRkBAAHfu3MHT0xMPDw9cXV2JiIjQ30/uac2aNePq1au4uLgU4giFEMVRiV2CaNu2LW3btgXAw8NDf6ugTp060alTJwAURWHgwIHA42WJ2bNnP3Mcd3d33N3dAbC3t+fdd9/N9vq4ceOy/azT6ShdujS9e/fO3wEJIUocg5gB/1NaWhr9+/fHzs6O2rVr59txY2Nj6devH87OznI3DCHEvyqxM+DnMTc3Z+vWrfl+3Fq1avHjjz/mefuSkAcsubxCvDz5y1FRScgDluIrxMuTvx4hhFCJFGAhhFCJFGAVFXYgu4SnC1G0GOSHcEVFYQeyF/cP/IQoaWQGLIQQKpECLIQQKjGYAhwZGYmXl1e25wICAli3bh3Lli3LdT8JZBdCFBSDKcC5sba2ZuzYsbm+LoHsQoiCIh/CAV5eXixZsoRp06Y9N5D96NGjrF+/HjMzM+rUqcPs2bP56aef+OGHH9DpdHz88ceEhoY+E/RuY2Oj8giFEEWRQRXgw4cP60N54HF2w/jx4wFISUnhyJEjbNmyBYADBw7oA9mdnZ2xsLAgMDCQbdu2YWlpyfz589m8eTOlS5fG2tqaFStWoCgK8+bN4969eyQmJlK+fHkpvkKIXBlUAW7Xrh1LlizR/xwQEKB//G+B7LGxsdSrV08fsvMkeL1Zs2b6mMucgt6FECI3BlWAn+fpQPb09HQ6d+7Mu+++my2QPSYmhocPH1K6dGn++usvfeF9OpB9wIABTJo0iUePHvHpp5+qNRwhRDFg8B/CPZGXQPZx48YxfPhwBg0axJ07dxg8ePAzx/ln0LsQQuTGYCrE0wHtT0yaNAmA/v37A+QpkP2fd7p4su/Tng56F0KI3MgMOB8VVNC7EKJkMpgZcGF40aD3wg5kl/B0IYoW+WtUUWEHskvxFaJokb9IIYRQiRRgFeV3HrDk/QpRvMgasIryOw9Y8n6FKF5kBiyEECqRAiyEECqRApyLESNGcOrUKQC0Wi2tWrVi9erV+tc9PDzy5bbuQgjDJQU4F46Ojhw9ehSAY8eO0bFjR/bu3QtAeno6169fx8HBQc0uCiGKOSnAuejQoYO+AO/duxdXV1cePHjAgwcPOH78OK1atcLX15dRo0bh4uKiT1mbOnUqY8aMwd3dnXv37qk5BCFEEScFOBeNGjXi0qVLKIrCkSNHeOONN2jfvj0HDx7kr7/+on79+jRv3pw1a9YQFhZGSEiIft927doREhJC2bJlVRyBEKKok6+h5cLIyAgHBwciIiKoXLkyZmZmdOrUiT179nDu3Dk+//xzgoKCOHz4MJaWlmi1Wv2+T2IqhRDieWQG/ByOjo588803vPnmmwC0atWKs2fPotPp2LVrF1ZWVixevJiRI0eSlpaGoijA42B2IYT4N1KAn6NDhw4cO3aMzp07A2BmZoaVlZV+OWLfvn0MHTqUmTNnUrt2bbl5pxDihcgSxHPUqFGD8+fPZ3tu+fLl+sfbt29/Zh9/f/8C75cQomSQGbAQQqhEZsAqyu88YMn7FaJ4kb9WFeV3HrAUXyGKF/mLFUIIlUgBVlF+5gFLFrAQxY+sAasoP/OAJQtYiOJHZsBCCKESKcBCCKESgy7AkZGRNGjQgB07dmR73sXFhalTp+b5OFFRUSxbtiy/uyeEKOEMugAD2NnZZSvA58+f59GjRy90jIYNGzJ27Nj87poQooQz+ALs4OBAfHw8Dx48AB5fXuzi4gI8DuN5wsvLi8jISC5fvoy7uzvDhg1jyJAhJCQkEBkZiZeXFwChoaH079+fvn378tVXXxX+gIQQxYbBF2AAJycnfv/9dxRF4dSpU7Ro0SLXbQ8ePEjTpk359ttvGTdunL5wA9y+fZtVq1axceNGtm3bhlarJTU1tTCGIIQohqQA83jN95dffuHIkSO0bt06x22eRE0OHDgQa2tr3n//fTZs2ICxsbF+m9jYWOrXr4+5uTkajYZJkyZRpkyZQhmDEKL4kQIM1KpVi4cPHxIcHEyfPn30z2dmZpKamopWq+XixYsA7N69m1atWrF+/Xp69uyZ7Uadtra2XLp0SR/OPn78eG7evFm4gxFCFBtyIcb/5+zszI8//kjdunWJjY0FYPjw4bi5uVGzZk2qV68OQJMmTfD29mbFihXodDqmTZtGSkoKABUqVGD06NEMGzYMjUZD165dsbGxUW1MQoiiTaM8+be1KHRRUVHM/sUiX471MlfCRUVF0bBhw3xp/2UYevtFoQ/SvrrtyxKEEEKoRJYgVJSfecCSBSxE8SN/sSrKzzxgKb5CFD/yVyuEECqRAiyEECqRAqyilw1kl/B1IUoG+RBORS8byC7h60KUDDIDFkIIlRTbGbC/vz9nzpwhKSmJtLQ0atWqRfny5bGwsKBNmzYMHDhQv+26deu4c+eOPrEMwMPDg1u3brFz5079c7///jvjxo1j9+7d1KxZM8998fLyYsGCBZiZmeXP4IQQBqHYFuAngelbt27l0qVLTJo0CYCjR4+ydOnSbAV427ZtfP311zke5+krYXbs2EGNGjVeuC9Llix54X2EEKLELUG0bt2a5ORkrl+/DsCpU6eoVKlSjjPad955h59//hmA+/fvk56eTqVKlfQ/f/jhhwwdOhR3d3cOHTrEuXPn8PDw0O//4YcfcvbsWbp160Z6ejoJCQm8//77eHh48P7775OQkFAIIxZCFFclrgDD48jI7du3A49nyO7u7jlu161bNyIiIlAUhd9++42ePXvqX1uxYgUdOnRgw4YNLF26lOnTp9OgQQO0Wi3Xr18nMTGRO3fu0KhRI/0+CxYswMPDg+DgYEaNGkVAQEDBDlQIUayVyAL87rvvsnPnTtLT0/nrr7/o2rVrjtuVKlWKhg0bcvz4cXbt2sVbb72lfy0mJoY2bdoAYGNjg6WlJbdv32bgwIH85z//4ccff6R///7ZjhcdHc0333yDh4cHX3/9Nbdv3y64QQohir1iuwb8PBUqVMDe3p7ly5fz1ltvYWKS+zB79+7NunXrsLa2zhaebm9vz9GjR2nUqBE3b97k/v37lCtXDmdnZzw9PTEyMmLNmjXZjmVnZ8fIkSNp2bIlMTExHDlypMDGKIQo/kpkAQYYNGgQo0eP5tdff33udh06dGDq1Kl8/vnn2Z7/8MMP8fHx4bfffiMtLY3Zs2djYmKCiYkJDg4OZGZmYmlpmW0fb29vZs6cSXp6OmlpaUyfPj3fxyWEKDkkD1hFL5sHnF8XYqidhWro7ReFPkj7kgcshBAGSQqwEEKopMSuARcHLxvILuHrQpQM8lesopcNZJfiK0TJIH/JQgihEinAKnqZPGDJAhai5JA1YBW9TB6wZAELUXLIDFgIIVQiBVgIIVSiyhJEZGQkEyZMoF69eiiKQmZmJsOHD8fZ2VmN7uTq7t277Nu3DxcXF7W7IoQogVRbA27Xrp0+yDw1NRUPDw/q1q2r+qWhTzt//jzh4eFSgIUQBaJIfAhXpkwZ3Nzc+PXXX/nxxx85duwY8DipbMSIEVy5cgVfX18yMjIwNzdnyZIlLFy4EGdnZzp16kRERAS//PIL/v7+vPXWW7Ro0YIrV67Qvn17Hjx4wKlTp6hbty6LFi0iISEBPz8/0tPTKVWqFHPmzCErK4tPP/2UqlWrEhsby+uvv86sWbMICgri3LlzbN68mRYtWuDv709WVhZ37txh5syZtGzZkq5du2JnZ4e9vT1//vknoaGhlCtXjo0bN5Kamsro0aNVPrtCiKKqSBRggIoVK7J69WoaNGjAli1byMzMZMiQIbRr144vv/ySDz74gE6dOrF7927Onj2b63GuX7/O+vXrqVy5Mm+88QahoaH4+fnRvXt37t+/rw9N79y5M4cOHSIgIAAvLy+uXLnCmjVrsLCwoEePHiQlJTFmzBhCQkJwc3Pjl19+wdvbmwYNGvDTTz+xdetWWrZsSUJCAlu3bqV8+fJYWlqyY8cOhg4dyvbt21m2bFkhnkEhRHFTZApwfHw8ffv2pUyZMmg0GkxNTWnWrBkxMTFcvnyZFi1aANC9e3cA/a2EAJ4OdCtXrhzVq1cHoHTp0tSrVw8AKysr0tPT9aHpq1evRlEUfVawra2tPl6ycuXKpKdnv0qtSpUqLF++HHNzc1JTU/Xbli9fnvLlywMwYMAAJk6cSJs2bahUqZL+9kZCCJGTIvEtiJSUFEJDQ7G0tNQvP2RkZHD8+HFq166Nvb09p0+fBmD79u0EBwdjZmZGUlISQLYZsUajeW5bdnZ2TJo0ieDgYGbNmqW/DVFO+xkZGaHTPb7wYd68eYwfP54FCxbw2muv6Yu+kdH/TmGNGjWwsrIiKCgo201BhRAiJ6rNgA8fPoyHhwdGRkZkZWUxbtw4nJycuHHjBm5ubmRkZNCzZ08aN27MlClTmDFjBitWrMDc3JxFixYRGxuLj48PP/30E3Xq1Mlzuy8Smm5ra0t0dDTr1q2jT58+fPLJJ1hbW1O1alXu3LmT4z6DBg1i7ty5LFq06EVPiRDCwEggez7buXMn0dHRfPLJJ/+67csEsufnlXBqh1EbevtFoQ/SvrrtF5k14JLgiy++IDIykqCgILW7IoQoBqQA56OJEye+0PYvkwcsWcBClBzyl6yil8kDluIrRMkhf81CCKESKcAqyksesOT/ClFyyRqwivKSByz5v0KUXDIDFkIIlUgBFkIIlZSoAjxixAhOnToFgFarpVWrVqxevVr/uoeHB1FRUXk6Vnp6OqGhoQXSTyGEgBJWgB0dHTl69CgAx44do2PHjuzduxd4XFCvX7+Og4NDno6VlJQkBVgIUaBK1IdwHTp0YPny5YwcOZK9e/fi6upKQEAADx484MyZM7zxxhv8+uuvrFu3DiMjI1q1asWkSZM4duwYCxYswMTEBAsLC5YuXUpQUBAXL15k2bJljBgxgunTp+vzH3x9fWnQoEG2LOD79+9jZmbG9evXSUxMxN/fn8aNG6t8RoQQRVmJmgE3atSIS5cuoSgKR44c4Y033qB9+/YcPHiQv/76C0dHRwIDA1m3bh2bNm3i5s2bHDhwgF27dtGrVy++//57Bg8ezP379xkzZgz16tVj7NixBAUF0a5dO4KDg5kzZw4zZ84EICEhgYCAAHx8fACoXr06a9aswcPDg82bN6t4JoQQxUGJmgEbGRnh4OBAREQElStXxszMjE6dOrFnzx7OnTtHly5dSE5O5oMPPgAe3wrp2rVrjBkzhqCgIEaMGIGNjQ1NmzZFq9XqjxsdHc3hw4fZuXMnAPfu3QOyZwED+lCPqlWr8t///rewhi2EKKZK1AwYHq8Df/PNN7z55psAtGrVirNnz6LT6ahZsybVqlVj7dq1BAcHM2zYMJo3b8727dvp168fwcHB1K9fny1btmTLArazs8PT05Pg4GC+/PJL+vTpA2TPAoZ/zyIWQoinlagZMDxeB/b19WXhwoUAmJmZYWVlRcOGDalQoQKenp54eHiQlZVFjRo16NWrF1qtFl9fXywsLDAyMmL27NlUrFiRjIwMFi1axJgxY5g+fTpbtmwhJSWFsWPHqjxKIURJIHnAKspLHnBBXgmndhaqobdfFPog7avbfolbghBCiOKixC1BFCd5yQOW/F8hSi75y1ZRXvKApfgKUXLJGrCKTpw4QalS/x5JKYQo3kxMTKhfv/4zz0sBFkIIlci/b4UQQiVSgIUQQiVSgIUQQiVSgIUQQiVSgIUQQiVSgIUQQiVyJZwKdDodM2fO5Pz585iZmTF37lxq165dIG3169cPS0tLAGrWrImbmxvz5s3D2NiYjh07Mnbs2ALpz8mTJwkICCA4OJirV68ydepUNBoN9evX57PPPsPIyIhly5axZ88eTExM8PHxoWnTprlu+6p9OHv2LB9++CF16tQBYPDgwTg7OxdIHzIyMvDx8eH69etotVo++ugj6tWrV2jnIKf2q1WrVmjjz8rKwtfXl8uXL6PRaJg1axalSpUqtPHn1H5mZmahjf+FKKLQ/fbbb4q3t7eiKIpy/PhxZcyYMQXSTlpamvLuu+9me65Pnz7K1atXFZ1Op7z//vvKmTNn8r0/K1euVHr37q24uroqiqIoH374oXL48GFFURTFz89P+f3335W///5b8fDwUHQ6nXL9+nWlf//+uW6bH33YsmWLsmbNmmzbFFQfwsLClLlz5yqKoih37txROnfuXKjnIKf2C3P8f/zxhzJ16lRFURTl8OHDypgxYwp1/Dm1X5jjfxGyBKGCY8eO6fOKmzdvzt9//10g7Zw7d45Hjx4xcuRIhg8fzpEjR9Bqtdja2qLRaOjYsSMHDx7M9/7Y2toSGBio//nJ7aAAOnXqpG+zY8eOaDQaqlevTlZWFsnJyTlumx99+Pvvv9mzZw9Dhw7Fx8eHlJSUAutDz549+eSTTwBQFAVjY+NCPQc5tV+Y4+/Rowdz5swBID4+Hmtr60Idf07tF+b4X4QUYBWkpKTolwUAjI2NyczMzPd2zM3NGTVqFGvWrGHWrFlMmzYNC4v/xV+WKVOGBw8e5Ht/3n77bUxM/re6pSiKPqw+tzafPJ/TtvnRh6ZNmzJlyhQ2bNhArVq1+PrrrwusD2XKlMHS0pKUlBTGjx/PhAkTCvUc5NR+YY4fHl966+3tzZw5c3BxcSn098A/2y/s8eeVFGAVWFpakpqaqv9Zp9NlKxb5pW7duvTp0weNRkPdunWxsrLi7t27+tdTU1OxtrYu8P48vX6WW5upqalYWVnluG1+eOutt2jSpIn+8dmzZwu0DwkJCQwfPpx3330XFxeXQj8H/2y/sMcPsGDBAn777Tf8/PxIT/9f8FRhvQeebr9jx46FPv68kAKsgpYtWxIREQE8DuR57bXXCqSdsLAw/P39Abh58yaPHj2idOnSXLt2DUVR2L9/P61bty7w/jRq1IjIyEgAIiIi9G3u378fnU5HfHw8Op2OChUq5Lhtfhg1ahSnTp0C4NChQzRu3LjA+nDr1i1GjhzJ5MmTGThwYKGfg5zaL8zx/+c//+Gbb74BwMLCAo1GQ5MmTQpt/Dm1P3bs2EIb/4uQMB4VPPnWQXR0NIqiMH/+fOzt7fO9Ha1Wy7Rp04iPj0ej0TBp0iSMjIyYP38+WVlZdOzYES8vrwLpT1xcHBMnTmTLli1cvnwZPz8/MjIysLOzY+7cuRgbGxMYGEhERAQ6nY5p06bRunXrXLd91T6cOXOGOXPmYGpqSqVKlZgzZw6WlpYF0oe5c+eyc+dO7Ozs9M9Nnz6duXPnFso5yKn9CRMmsGjRokIZ/8OHD5k2bRq3bt0iMzOT0aNHY29vX2jvgZzar1atWqH9/l+EFGAhhFCJLEEIIYRKpAALIYRKpAALIYRKpAALIYRKpAALIYRKpAALUcTcvXuXn376Se1uiEIgBViIIub8+fOEh4er3Q1RCCSOUohXkJaWpr/Y5UkMZEhICHFxcWRlZfHee+/h7OyMh4cHM2fOxN7enk2bNnHr1i369evHp59+StWqVYmNjeX1119n1qxZBAUFce7cOTZv3oybm5vaQxQFSAqwEK8gJCSEGjVqsGTJEq5cucIvv/xChQoVCAgIICUlhf79+9OuXbtc979y5Qpr1qzBwsKCHj16kJSUxJgxYwgJCZHiawBkCUKIV3Dp0iWaN28OQJ06dUhKSqJNmzbA49Ale3t7YmNjs+3z9MWntra2WFpaYmxsTOXKlbOF1oiSTwqwEK/A3t6e06dPAxAbG8uOHTs4evQo8Dh2NDo6mpo1a2JmZkZSUhIAZ8+e1e//JPbwaUZGRuh0ukLovVCbFGAhXoG7uztxcXEMGzaMKVOmsHr1au7evcvgwYMZPnw4Y8eOpWLFigwfPpxZs2YxatQosrKynntMW1tboqOjWbduXeEMQqhGwniEEEIlMgMWQgiVSAEWQgiVSAEWQgiVSAEWQgiVSAEWQgiVSAEWQgiVSAEWQgiV/D8PuWdJsfQMEQAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# visualizing genre column\n",
    "sns.catplot(y = 'Genre', data = df, kind = 'count', \n",
    "            order = df['Genre'].value_counts().index,\n",
    "            color = '#4287f5')\n",
    "plt.title('genre column distribution')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- we can notice from the above visual that `Drama` genre is the most frequent genre in our dataset and has appeared more than 14% of the times among 19 other genres."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Q2: What `genres` has highest `votes`?\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# visualizing vote_average column\n",
    "sns.catplot(y = 'Vote_Average', data = df, kind = 'count', \n",
    "            order = df['Vote_Average'].value_counts().index,\n",
    "            color = '#4287f5')\n",
    "plt.title('votes destribution')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(6520, 6)\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Release_Date</th>\n",
       "      <th>Title</th>\n",
       "      <th>Popularity</th>\n",
       "      <th>Vote_Count</th>\n",
       "      <th>Vote_Average</th>\n",
       "      <th>Genre</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2021</td>\n",
       "      <td>Spider-Man: No Way Home</td>\n",
       "      <td>5083.954</td>\n",
       "      <td>8940</td>\n",
       "      <td>popular</td>\n",
       "      <td>Action</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2021</td>\n",
       "      <td>Spider-Man: No Way Home</td>\n",
       "      <td>5083.954</td>\n",
       "      <td>8940</td>\n",
       "      <td>popular</td>\n",
       "      <td>Adventure</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2021</td>\n",
       "      <td>Spider-Man: No Way Home</td>\n",
       "      <td>5083.954</td>\n",
       "      <td>8940</td>\n",
       "      <td>popular</td>\n",
       "      <td>Science Fiction</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2022</td>\n",
       "      <td>The Batman</td>\n",
       "      <td>3827.658</td>\n",
       "      <td>1151</td>\n",
       "      <td>popular</td>\n",
       "      <td>Crime</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>2022</td>\n",
       "      <td>The Batman</td>\n",
       "      <td>3827.658</td>\n",
       "      <td>1151</td>\n",
       "      <td>popular</td>\n",
       "      <td>Mystery</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Release_Date                    Title  Popularity  Vote_Count Vote_Average  \\\n",
       "0          2021  Spider-Man: No Way Home    5083.954        8940      popular   \n",
       "1          2021  Spider-Man: No Way Home    5083.954        8940      popular   \n",
       "2          2021  Spider-Man: No Way Home    5083.954        8940      popular   \n",
       "3          2022               The Batman    3827.658        1151      popular   \n",
       "4          2022               The Batman    3827.658        1151      popular   \n",
       "\n",
       "             Genre  \n",
       "0           Action  \n",
       "1        Adventure  \n",
       "2  Science Fiction  \n",
       "3            Crime  \n",
       "4          Mystery  "
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# saperating popular movies\n",
    "popular_movies = df[df['Vote_Average'] == 'popular']\n",
    "print(popular_movies.shape)\n",
    "popular_movies.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "it looks like we have 25.5% of our dataset with popular vote (6520 rows). let's visualize it more wwith coresponding `genre`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# visualizing genre againest vote_average\n",
    "sns.catplot(x = 'Genre', data = popular_movies,\n",
    "            kind = 'count', order = popular_movies['Genre'].value_counts().index,\n",
    "            color = '#4287f5')\n",
    "plt.title('popular genres distribution')\n",
    "plt.xticks(rotation=90)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- as we can see, `Drama` again gets the highest popularity among fans by being having more than 18.5% of movies popularities."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Q3: What movie got the highest `popularity`? what's its `genre`?\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Release_Date</th>\n",
       "      <th>Title</th>\n",
       "      <th>Popularity</th>\n",
       "      <th>Vote_Count</th>\n",
       "      <th>Vote_Average</th>\n",
       "      <th>Genre</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2021</td>\n",
       "      <td>Spider-Man: No Way Home</td>\n",
       "      <td>5083.954</td>\n",
       "      <td>8940</td>\n",
       "      <td>popular</td>\n",
       "      <td>Action</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2021</td>\n",
       "      <td>Spider-Man: No Way Home</td>\n",
       "      <td>5083.954</td>\n",
       "      <td>8940</td>\n",
       "      <td>popular</td>\n",
       "      <td>Adventure</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2021</td>\n",
       "      <td>Spider-Man: No Way Home</td>\n",
       "      <td>5083.954</td>\n",
       "      <td>8940</td>\n",
       "      <td>popular</td>\n",
       "      <td>Science Fiction</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Release_Date                    Title  Popularity  Vote_Count Vote_Average  \\\n",
       "0          2021  Spider-Man: No Way Home    5083.954        8940      popular   \n",
       "1          2021  Spider-Man: No Way Home    5083.954        8940      popular   \n",
       "2          2021  Spider-Man: No Way Home    5083.954        8940      popular   \n",
       "\n",
       "             Genre  \n",
       "0           Action  \n",
       "1        Adventure  \n",
       "2  Science Fiction  "
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# checking max popularity in dataset\n",
    "df[df['Popularity'] == df['Popularity'].max()]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- we can see that `Spider-Man: No Way Home` has the highest popularity rate in our dataset and it has genres of `Action`, `Adventure` and `Sience Fiction`. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Q4: Which year has the most filmmed movies?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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48aN47rrruPOO+9sFopXUm5uLvPmzWPcuHF873vfcz8fXzdt2jSysrIYP348AQEB3HLLLYwfPx6r1UpUVBTjxo1j8+bN37iewYMHM3/+fE6fPk1kZCS//OUvAZgxYwZZWVns3LmTyMjIZoeNxo4dy9KlS5v9byA8PJznn3+exYsX09jYiMVioaCggIiICP70pz9doWdFvolF1/MXETEf7flLl5Kamordbvd438aNGwkODvbxjER6Ju35i4iYkL7wFRExIa8O+xw4cIDly5dTVFTkHnv99dcpLi52/8KgpKSELVu24O/vz4wZMxg7dix1dXU8+eSTNDY20q9fPwoKCrjqqqs81rZl//79BAYGtlnncDi8qusO1EvX01P6APXSFX0XfTgcDkaMGNHyDqMN69atM8aPH29MmjTJPVZdXW089NBD7rFTp04Z48ePNxwOh3Hu3Dn3vxcvXmxs3brVMAzDWLt2rfHiiy+2WtuWP//5z23WtKeuO1AvXU9P6cMw1EtX9F300dpjtnnYZ+DAgaxYscJ9+8yZMxQWFjJ//nz32MGDBxk5ciQBAQGEhIQwcOBAjhw5QlVVlfusxYSEBHbv3t1qrYiI+E6bh30SExPdZ2s6nU4WLFjAvHnzmv3XpKGhgZCQEPftoKAgGhoamo0HBQVRX1/fam1bHA4HNputzbrGxkav6roD9dL19JQ+QL10Rb7so10/9ayurqampoannnoKh8PB//zP/5Cfn88//MM/NPt5nt1uJyQkhODgYOx2O71798ZutxMaGuoe+9vatgQGBjJ06NA262w2m1d13YF66Xp6Sh+gXrqi76KP1j5M2vVrn2HDhvHGG29QVFREYWEhN998MwsWLGDYsGFUVVXhcDior6/n+PHjDB48mOjoaPcf2igvL+f2229vtVZERHznipzkdfl63qmpqRiGwezZswkMDGTGjBlkZmZSUlJC3759efbZZ+nTp4/HWhER8R2vwn/AgAGUlJR849jkyZOZPHlys5prr73W43W4PdWKiIjv6CQvERETUviLiJiQwl9ExIQU/iIiXmhscn7n62jtZ57fxbp1SWcRES/07mXlB1lvdMq6P3r6viv+mNrzFxExIYW/iIgJKfxFRExI4S8iYkIKfxERE1L4i4iYkMJfRMSEFP4iIiak8BcRMSGFv4iICSn8RURMSOEvImJCCn8R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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "df['Release_Date'].hist()\n",
    "plt.title('Release_Date column distribution')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- we can see from the above plot that year `2020` has the highest filmming rate in our dataset."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "___\n",
    "## Conclusion\n",
    "#### Q1: What is the most frequent `genre` in the dataset?\n",
    "`Drama` genre is the most frequent genre in our dataset and has appeared more than 14% of the times among 19 other genres.\n",
    "\n",
    "#### Q2: What `genres` has highest `votes`?\n",
    "we have 25.5% of our dataset with popular vote (6520 rows).\n",
    "`Drama` again gets the highest popularity among fans by being having more than 18.5% of movies popularities.\n",
    "\n",
    "#### Q3: What movie got the highest `popularity`? what's its `genre`?\n",
    "`Spider-Man: No Way Home` has the highest popularity rate in our dataset and it has genres of `Action`, `Adventure` and `Sience Fiction`. \n",
    "\n",
    "#### Q4: Which year has the most filmmed movies?\n",
    "year `2020` has the highest filmming rate in our dataset."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

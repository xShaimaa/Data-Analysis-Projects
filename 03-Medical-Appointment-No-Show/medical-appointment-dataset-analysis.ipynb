{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<a href=\"https://www.kaggle.com/code/xshaimaa/medical-appointment-dataset-analysis\" target=\"_blank\"><img align=\"left\" alt=\"Kaggle\" title=\"Open in Kaggle\" src=\"https://kaggle.com/static/images/open-in-kaggle.svg\"></a>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Dataset Description \n",
    "A person makes a doctor appointment, receives all the instructions and no-show. Who to blame?\n",
    "This dataset collects information from 100k medical appointments in Brazil and is focused on the question of whether or not patients show up for their appointment. A number of characteristics about the patient are included in each row.\n",
    " \n",
    " \n",
    "## Columns Description\n",
    "1. `PatientId`: Identification of a patient.\n",
    "2. `AppointmentID`: Identification of each appointment.\n",
    "3. `Gender`: Male or Female.\n",
    "4. `AppointmentDay`: The day of the actuall appointment, when they have to visit the doctor.\n",
    "5. `ScheduledDay`: The day someone called or registered the appointment, this is before appointment of course.\n",
    "6. `Age`: How old is the patient.\n",
    "7. `Neighbourhood`: Where the appointment takes place.\n",
    "8. `Scholarship`: True of False, indicates whether or not the patient is enrolled in Brasilian welfare program Bolsa Família.\n",
    "9. `Hipertension`: True or False.\n",
    "10. `Diabetes`: True or False.\n",
    "11. `Alcoholism`: True or False.\n",
    "12. `Handcap`: True or False.\n",
    "13. `SMS_received`: 1 or more messages sent to the patient.\n",
    "14. `No-show`: True (if the patient did not show up), or False (if the patient did show up).\n",
    "\n",
    "\n",
    "## EDA Questions\n",
    "### Q1: How often do men go to hospitals compared to women? Which of them is more likely to show up?\n",
    "### Q2: Does recieving an SMS as a reminder affect whether or not a patient may show up? is it correlated with number of days before the appointment?\n",
    "### Q3: Does having a scholarship affects showing up on a hospital appointment? What are the age groups affected by this?\n",
    "### Q4: Does having certain deseases affect whather or not a patient may show up to their appointment? is it affected by gender?\n",
    "___"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Environment set-up"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5"
   },
   "outputs": [],
   "source": [
    "# importing lib.\n",
    "import numpy as np \n",
    "import pandas as pd \n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# getting the csv file directory\n",
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
    "## Data Wrangling\n",
    "\n",
    "in this section, we'd load our data from a CSV file to a pandas dataframe, and then take a quick dive into exploring our dataset in details."
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
       "      <th>PatientId</th>\n",
       "      <th>AppointmentID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>ScheduledDay</th>\n",
       "      <th>AppointmentDay</th>\n",
       "      <th>Age</th>\n",
       "      <th>Neighbourhood</th>\n",
       "      <th>Scholarship</th>\n",
       "      <th>Hipertension</th>\n",
       "      <th>Diabetes</th>\n",
       "      <th>Alcoholism</th>\n",
       "      <th>Handcap</th>\n",
       "      <th>SMS_received</th>\n",
       "      <th>No-show</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2.987250e+13</td>\n",
       "      <td>5642903</td>\n",
       "      <td>F</td>\n",
       "      <td>2016-04-29T18:38:08Z</td>\n",
       "      <td>2016-04-29T00:00:00Z</td>\n",
       "      <td>62</td>\n",
       "      <td>JARDIM DA PENHA</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5.589978e+14</td>\n",
       "      <td>5642503</td>\n",
       "      <td>M</td>\n",
       "      <td>2016-04-29T16:08:27Z</td>\n",
       "      <td>2016-04-29T00:00:00Z</td>\n",
       "      <td>56</td>\n",
       "      <td>JARDIM DA PENHA</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>4.262962e+12</td>\n",
       "      <td>5642549</td>\n",
       "      <td>F</td>\n",
       "      <td>2016-04-29T16:19:04Z</td>\n",
       "      <td>2016-04-29T00:00:00Z</td>\n",
       "      <td>62</td>\n",
       "      <td>MATA DA PRAIA</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>8.679512e+11</td>\n",
       "      <td>5642828</td>\n",
       "      <td>F</td>\n",
       "      <td>2016-04-29T17:29:31Z</td>\n",
       "      <td>2016-04-29T00:00:00Z</td>\n",
       "      <td>8</td>\n",
       "      <td>PONTAL DE CAMBURI</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>8.841186e+12</td>\n",
       "      <td>5642494</td>\n",
       "      <td>F</td>\n",
       "      <td>2016-04-29T16:07:23Z</td>\n",
       "      <td>2016-04-29T00:00:00Z</td>\n",
       "      <td>56</td>\n",
       "      <td>JARDIM DA PENHA</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      PatientId  AppointmentID Gender          ScheduledDay  \\\n",
       "0  2.987250e+13        5642903      F  2016-04-29T18:38:08Z   \n",
       "1  5.589978e+14        5642503      M  2016-04-29T16:08:27Z   \n",
       "2  4.262962e+12        5642549      F  2016-04-29T16:19:04Z   \n",
       "3  8.679512e+11        5642828      F  2016-04-29T17:29:31Z   \n",
       "4  8.841186e+12        5642494      F  2016-04-29T16:07:23Z   \n",
       "\n",
       "         AppointmentDay  Age      Neighbourhood  Scholarship  Hipertension  \\\n",
       "0  2016-04-29T00:00:00Z   62    JARDIM DA PENHA            0             1   \n",
       "1  2016-04-29T00:00:00Z   56    JARDIM DA PENHA            0             0   \n",
       "2  2016-04-29T00:00:00Z   62      MATA DA PRAIA            0             0   \n",
       "3  2016-04-29T00:00:00Z    8  PONTAL DE CAMBURI            0             0   \n",
       "4  2016-04-29T00:00:00Z   56    JARDIM DA PENHA            0             1   \n",
       "\n",
       "   Diabetes  Alcoholism  Handcap  SMS_received No-show  \n",
       "0         0           0        0             0      No  \n",
       "1         0           0        0             0      No  \n",
       "2         0           0        0             0      No  \n",
       "3         0           0        0             0      No  \n",
       "4         1           0        0             0      No  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# loading dataset from csv file and showing its first 5 rows\n",
    "df = pd.read_csv('noshowappointments-kagglev2-may-2016.csv')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We'll move next into exploring our dataset by going through its data types, NaNs or duplicated rows, and any columns that may need to be dropped or parsed."
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
      "RangeIndex: 110527 entries, 0 to 110526\n",
      "Data columns (total 14 columns):\n",
      " #   Column          Non-Null Count   Dtype  \n",
      "---  ------          --------------   -----  \n",
      " 0   PatientId       110527 non-null  float64\n",
      " 1   AppointmentID   110527 non-null  int64  \n",
      " 2   Gender          110527 non-null  object \n",
      " 3   ScheduledDay    110527 non-null  object \n",
      " 4   AppointmentDay  110527 non-null  object \n",
      " 5   Age             110527 non-null  int64  \n",
      " 6   Neighbourhood   110527 non-null  object \n",
      " 7   Scholarship     110527 non-null  int64  \n",
      " 8   Hipertension    110527 non-null  int64  \n",
      " 9   Diabetes        110527 non-null  int64  \n",
      " 10  Alcoholism      110527 non-null  int64  \n",
      " 11  Handcap         110527 non-null  int64  \n",
      " 12  SMS_received    110527 non-null  int64  \n",
      " 13  No-show         110527 non-null  object \n",
      "dtypes: float64(1), int64(8), object(5)\n",
      "memory usage: 11.8+ MB\n"
     ]
    }
   ],
   "source": [
    "# viewing main info about df\n",
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- we can notice there are no NaNs at all in our data\n",
    "- `PatientId` and `AppointmentId` columns wouldn't be helpful during analysis.\n",
    "- `ScheduledDay` and `AppointmentDay` needs to be casted to date data type.\n",
    "- we may append a new column for days until appointment.\n",
    "- `Gender` needs to be converted into a categoy type\n",
    "- `Scholarship` `Hipertension` `Diabetes` `Alcoholism` `Handcap` better be boolean data type.\n",
    "- `No-show` needs to be parsed and casted to boolean too."
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
       "0"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# checking for duplicates\n",
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
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "PatientId          62299\n",
       "AppointmentID     110527\n",
       "Gender                 2\n",
       "ScheduledDay      103549\n",
       "AppointmentDay        27\n",
       "Age                  104\n",
       "Neighbourhood         81\n",
       "Scholarship            2\n",
       "Hipertension           2\n",
       "Diabetes               2\n",
       "Alcoholism             2\n",
       "Handcap                5\n",
       "SMS_received           2\n",
       "No-show                2\n",
       "dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# exploring the unique values of each column\n",
    "df.nunique()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- `Handcap` and `Age` columns has inconsistant unique values.\n",
    "- `SMS_received` would be casted to boolean data type."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    108286\n",
       "1      2042\n",
       "2       183\n",
       "3        13\n",
       "4         3\n",
       "Name: Handcap, dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# exploring handcap values\n",
    "df['Handcap'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- we'd be only intrested in rows with `0` or `1` values."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count    110527.000000\n",
       "mean         37.088874\n",
       "std          23.110205\n",
       "min          -1.000000\n",
       "25%          18.000000\n",
       "50%          37.000000\n",
       "75%          55.000000\n",
       "max         115.000000\n",
       "Name: Age, dtype: float64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# exploring age column distribution\n",
    "df['Age'].describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- `Age` column would need to be handled."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Exploration Summery\n",
    "1. our dataset consists of 110527 rows with 14 columns, and has no NaNs nor duplicated values.\n",
    "2. `PatientId` and `AppointmentId` columns wouldn't be helpful during analysis.\n",
    "3. `ScheduledDay` and `AppointmentDay` needs to be casted to date data type.\n",
    "4. we may append a new column for days until appointment.\n",
    "5. `Gender` needs to be casted into a categoy type\n",
    "6. `Scholarship`, `Hipertension`, `Diabetes`, `Alcoholism` and `SMS_recieved` better be boolean data type.\n",
    "7. `No-show` column needs to be parsed and asted to boolean type.\n",
    "8. `Handcap` colume needs to be cleaned to have only `0` and `1` values.\n",
    "9. `Age` columns has inconsistant unique values that needs to be handled.\n",
    "___"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data Cleaning\n",
    "in this section, we'd perform some operations on our dataset based on the previous findings to make our analysis more accurate and clear."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "___\n",
    "**Dropping `PatientId` and `AppointmentId` columns**"
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
       "Index(['Gender', 'ScheduledDay', 'AppointmentDay', 'Age', 'Neighbourhood',\n",
       "       'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap',\n",
       "       'SMS_received', 'No-show'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# dropping columns and validating changes\n",
    "df.drop(['PatientId', 'AppointmentID'], axis = 1, inplace = True)\n",
    "df.columns"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "___\n",
    "**Handling `date` data type**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<bound method Series.unique of 0         2016-04-29T00:00:00Z\n",
       "1         2016-04-29T00:00:00Z\n",
       "2         2016-04-29T00:00:00Z\n",
       "3         2016-04-29T00:00:00Z\n",
       "4         2016-04-29T00:00:00Z\n",
       "                  ...         \n",
       "110522    2016-06-07T00:00:00Z\n",
       "110523    2016-06-07T00:00:00Z\n",
       "110524    2016-06-07T00:00:00Z\n",
       "110525    2016-06-07T00:00:00Z\n",
       "110526    2016-06-07T00:00:00Z\n",
       "Name: AppointmentDay, Length: 110527, dtype: object>"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.AppointmentDay.unique"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "it looks like all hours are set to 00:00:00, so we would want to extract onl the year, month and day data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "AppointmentDay    datetime64[ns]\n",
      "ScheduledDay      datetime64[ns]\n",
      "dtype: object\n"
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
       "      <th>Gender</th>\n",
       "      <th>ScheduledDay</th>\n",
       "      <th>AppointmentDay</th>\n",
       "      <th>Age</th>\n",
       "      <th>Neighbourhood</th>\n",
       "      <th>Scholarship</th>\n",
       "      <th>Hipertension</th>\n",
       "      <th>Diabetes</th>\n",
       "      <th>Alcoholism</th>\n",
       "      <th>Handcap</th>\n",
       "      <th>SMS_received</th>\n",
       "      <th>No-show</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>F</td>\n",
       "      <td>2016-04-29</td>\n",
       "      <td>2016-04-29</td>\n",
       "      <td>62</td>\n",
       "      <td>JARDIM DA PENHA</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>M</td>\n",
       "      <td>2016-04-29</td>\n",
       "      <td>2016-04-29</td>\n",
       "      <td>56</td>\n",
       "      <td>JARDIM DA PENHA</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>F</td>\n",
       "      <td>2016-04-29</td>\n",
       "      <td>2016-04-29</td>\n",
       "      <td>62</td>\n",
       "      <td>MATA DA PRAIA</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>F</td>\n",
       "      <td>2016-04-29</td>\n",
       "      <td>2016-04-29</td>\n",
       "      <td>8</td>\n",
       "      <td>PONTAL DE CAMBURI</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>F</td>\n",
       "      <td>2016-04-29</td>\n",
       "      <td>2016-04-29</td>\n",
       "      <td>56</td>\n",
       "      <td>JARDIM DA PENHA</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>No</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Gender ScheduledDay AppointmentDay  Age      Neighbourhood  Scholarship  \\\n",
       "0      F   2016-04-29     2016-04-29   62    JARDIM DA PENHA            0   \n",
       "1      M   2016-04-29     2016-04-29   56    JARDIM DA PENHA            0   \n",
       "2      F   2016-04-29     2016-04-29   62      MATA DA PRAIA            0   \n",
       "3      F   2016-04-29     2016-04-29    8  PONTAL DE CAMBURI            0   \n",
       "4      F   2016-04-29     2016-04-29   56    JARDIM DA PENHA            0   \n",
       "\n",
       "   Hipertension  Diabetes  Alcoholism  Handcap  SMS_received No-show  \n",
       "0             1         0           0        0             0      No  \n",
       "1             0         0           0        0             0      No  \n",
       "2             0         0           0        0             0      No  \n",
       "3             0         0           0        0             0      No  \n",
       "4             1         1           0        0             0      No  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# extracting only day, month and year values\n",
    "df['ScheduledDay'] = df['ScheduledDay'].str[:10]\n",
    "df['AppointmentDay'] = df['AppointmentDay'].str[:10]\n",
    "\n",
    "# changing data type\n",
    "df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])\n",
    "df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])\n",
    "\n",
    "# confirming changes\n",
    "print(df[['AppointmentDay', 'ScheduledDay']].dtypes)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now, we'd move into appending a new column that holds number of days to the appointment."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "# making new due days column\n",
    "df['due-days'] = df['AppointmentDay'] - df['ScheduledDay']\n",
    "\n",
    "# converting data type \n",
    "df['due-days'] = df['due-days'].dt.days\n",
    "\n",
    "# drop sch and appoint col\n",
    "df.drop(['AppointmentDay', 'ScheduledDay'], axis = 1, inplace = True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We'll move into exploring this new column."
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
       "count    110527.000000\n",
       "mean         10.183702\n",
       "std          15.254996\n",
       "min          -6.000000\n",
       "25%           0.000000\n",
       "50%           4.000000\n",
       "75%          15.000000\n",
       "max         179.000000\n",
       "Name: due-days, dtype: float64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# viewing summery statistics\n",
    "df['due-days'].describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We seem to have some negative values here, we'll drop them."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
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
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Neighbourhood</th>\n",
       "      <th>Scholarship</th>\n",
       "      <th>Hipertension</th>\n",
       "      <th>Diabetes</th>\n",
       "      <th>Alcoholism</th>\n",
       "      <th>Handcap</th>\n",
       "      <th>SMS_received</th>\n",
       "      <th>No-show</th>\n",
       "      <th>due-days</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>27033</th>\n",
       "      <td>M</td>\n",
       "      <td>38</td>\n",
       "      <td>RESISTÊNCIA</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>Yes</td>\n",
       "      <td>-1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>55226</th>\n",
       "      <td>F</td>\n",
       "      <td>19</td>\n",
       "      <td>SANTO ANTÔNIO</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>Yes</td>\n",
       "      <td>-1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>64175</th>\n",
       "      <td>F</td>\n",
       "      <td>22</td>\n",
       "      <td>CONSOLAÇÃO</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>Yes</td>\n",
       "      <td>-1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>71533</th>\n",
       "      <td>F</td>\n",
       "      <td>81</td>\n",
       "      <td>SANTO ANTÔNIO</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>Yes</td>\n",
       "      <td>-6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>72362</th>\n",
       "      <td>M</td>\n",
       "      <td>7</td>\n",
       "      <td>TABUAZEIRO</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>Yes</td>\n",
       "      <td>-1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      Gender  Age  Neighbourhood  Scholarship  Hipertension  Diabetes  \\\n",
       "27033      M   38    RESISTÊNCIA            0             0         0   \n",
       "55226      F   19  SANTO ANTÔNIO            0             0         0   \n",
       "64175      F   22     CONSOLAÇÃO            0             0         0   \n",
       "71533      F   81  SANTO ANTÔNIO            0             0         0   \n",
       "72362      M    7     TABUAZEIRO            0             0         0   \n",
       "\n",
       "       Alcoholism  Handcap  SMS_received No-show  due-days  \n",
       "27033           0        1             0     Yes        -1  \n",
       "55226           0        1             0     Yes        -1  \n",
       "64175           0        0             0     Yes        -1  \n",
       "71533           0        0             0     Yes        -6  \n",
       "72362           0        0             0     Yes        -1  "
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# viewing negative days values\n",
    "df[df['due-days'] < 0 ]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count    110522.000000\n",
       "mean         10.184253\n",
       "std          15.255115\n",
       "min           0.000000\n",
       "25%           0.000000\n",
       "50%           4.000000\n",
       "75%          15.000000\n",
       "max         179.000000\n",
       "Name: due-days, dtype: float64"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# dropping these values and confirming changes\n",
    "df.drop(df[df['due-days'] < 0].index, inplace = True)\n",
    "df['due-days'].describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "___\n",
    "**Converting `Gender` and `No-show` to categorical variables**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "CategoricalDtype(categories=['F', 'M'], ordered=False)"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# converting column and confirming changes\n",
    "df['Gender'] = df['Gender'].astype('category')\n",
    "\n",
    "df['Gender'].dtypes"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "___\n",
    "**Converting `Scholarship`, `Hipertension`, `Diabetes`, `Alcoholism`, `Handcap` and `SMS_recieved` to boolean data type**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Scholarship     bool\n",
       "Hipertension    bool\n",
       "Diabetes        bool\n",
       "Alcoholism      bool\n",
       "SMS_received    bool\n",
       "dtype: object"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# converting columns to bool and confirming changes\n",
    "cols = ['Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'SMS_received']\n",
    "df[cols] = df[cols].astype('bool')\n",
    "df[cols].dtypes"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "___\n",
    "**Parsing and casting `No-show` column**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "dtype('bool')"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# mapping alues to be more familiar\n",
    "df.loc[df['No-show'] == 'Yes', 'No-show'] = 0\n",
    "df.loc[df['No-show'] == 'No', 'No-show'] = 1\n",
    "\n",
    "# casting dt type and confirming changes\n",
    "df['No-show'] = df['No-show'].astype(bool)\n",
    "df['No-show'].dtypes"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "___\n",
    "**Cleaning `Handcap` column**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
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
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Neighbourhood</th>\n",
       "      <th>Scholarship</th>\n",
       "      <th>Hipertension</th>\n",
       "      <th>Diabetes</th>\n",
       "      <th>Alcoholism</th>\n",
       "      <th>Handcap</th>\n",
       "      <th>SMS_received</th>\n",
       "      <th>No-show</th>\n",
       "      <th>due-days</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>946</th>\n",
       "      <td>M</td>\n",
       "      <td>94</td>\n",
       "      <td>BELA VISTA</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>2</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>15</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1665</th>\n",
       "      <td>M</td>\n",
       "      <td>64</td>\n",
       "      <td>SANTA MARTHA</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>2</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1666</th>\n",
       "      <td>M</td>\n",
       "      <td>64</td>\n",
       "      <td>SANTA MARTHA</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>2</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>30</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2071</th>\n",
       "      <td>M</td>\n",
       "      <td>64</td>\n",
       "      <td>SANTA MARTHA</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>2</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2091</th>\n",
       "      <td>F</td>\n",
       "      <td>11</td>\n",
       "      <td>ANDORINHAS</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>2</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>108376</th>\n",
       "      <td>F</td>\n",
       "      <td>44</td>\n",
       "      <td>ROMÃO</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>2</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>109484</th>\n",
       "      <td>M</td>\n",
       "      <td>64</td>\n",
       "      <td>DA PENHA</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>2</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>109733</th>\n",
       "      <td>F</td>\n",
       "      <td>34</td>\n",
       "      <td>JUCUTUQUARA</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>2</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>109975</th>\n",
       "      <td>M</td>\n",
       "      <td>39</td>\n",
       "      <td>PRAIA DO SUÁ</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>2</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>110107</th>\n",
       "      <td>F</td>\n",
       "      <td>44</td>\n",
       "      <td>RESISTÊNCIA</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>2</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>199 rows × 11 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       Gender  Age Neighbourhood  Scholarship  Hipertension  Diabetes  \\\n",
       "946         M   94    BELA VISTA        False          True      True   \n",
       "1665        M   64  SANTA MARTHA        False          True     False   \n",
       "1666        M   64  SANTA MARTHA        False          True     False   \n",
       "2071        M   64  SANTA MARTHA        False          True     False   \n",
       "2091        F   11    ANDORINHAS        False         False     False   \n",
       "...       ...  ...           ...          ...           ...       ...   \n",
       "108376      F   44         ROMÃO        False          True      True   \n",
       "109484      M   64      DA PENHA        False          True      True   \n",
       "109733      F   34   JUCUTUQUARA        False         False     False   \n",
       "109975      M   39  PRAIA DO SUÁ         True         False     False   \n",
       "110107      F   44   RESISTÊNCIA        False         False     False   \n",
       "\n",
       "        Alcoholism  Handcap  SMS_received  No-show  due-days  \n",
       "946          False        2          True     True        15  \n",
       "1665          True        2          True     True        30  \n",
       "1666          True        2         False     True        30  \n",
       "2071          True        2         False     True         0  \n",
       "2091         False        2         False     True         0  \n",
       "...            ...      ...           ...      ...       ...  \n",
       "108376       False        2          True     True         6  \n",
       "109484       False        2         False     True         2  \n",
       "109733       False        2          True     True         4  \n",
       "109975       False        2          True     True         4  \n",
       "110107       False        2          True     True         4  \n",
       "\n",
       "[199 rows x 11 columns]"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# viewing rows with values of handcap > 1\n",
    "df[df['Handcap'] > 1]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We have 199 rows with inconsistant values, we'd replace them with 1 to treat them as beeing handcaped"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([False,  True])"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# filling the bigger values with 1\n",
    "df.loc[df['Handcap'].isin([2, 3, 4]), 'Handcap'] = 1\n",
    "\n",
    "# casting type and confirming changes\n",
    "df['Handcap'] = df['Handcap'].astype('bool')\n",
    "df['Handcap'].unique()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "___\n",
    "**Cleaning `Age` column**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
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
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Neighbourhood</th>\n",
       "      <th>Scholarship</th>\n",
       "      <th>Hipertension</th>\n",
       "      <th>Diabetes</th>\n",
       "      <th>Alcoholism</th>\n",
       "      <th>Handcap</th>\n",
       "      <th>SMS_received</th>\n",
       "      <th>No-show</th>\n",
       "      <th>due-days</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>99832</th>\n",
       "      <td>F</td>\n",
       "      <td>-1</td>\n",
       "      <td>ROMÃO</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      Gender  Age Neighbourhood  Scholarship  Hipertension  Diabetes  \\\n",
       "99832      F   -1         ROMÃO        False         False     False   \n",
       "\n",
       "       Alcoholism  Handcap  SMS_received  No-show  due-days  \n",
       "99832       False    False         False     True         0  "
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#exploring values below 0\n",
    "df[df['Age'] < 0]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- we have one value with negative age, so we will drop it"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
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
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Neighbourhood</th>\n",
       "      <th>Scholarship</th>\n",
       "      <th>Hipertension</th>\n",
       "      <th>Diabetes</th>\n",
       "      <th>Alcoholism</th>\n",
       "      <th>Handcap</th>\n",
       "      <th>SMS_received</th>\n",
       "      <th>No-show</th>\n",
       "      <th>due-days</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "Empty DataFrame\n",
       "Columns: [Gender, Age, Neighbourhood, Scholarship, Hipertension, Diabetes, Alcoholism, Handcap, SMS_received, No-show, due-days]\n",
       "Index: []"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# dropping row with negative age and confirming changes\n",
    "df.drop(df[df['Age'] < 0].index, inplace = True)\n",
    "df[df['Age'] < 0]"
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
   "execution_count": 24,
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
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Neighbourhood</th>\n",
       "      <th>Scholarship</th>\n",
       "      <th>Hipertension</th>\n",
       "      <th>Diabetes</th>\n",
       "      <th>Alcoholism</th>\n",
       "      <th>Handcap</th>\n",
       "      <th>SMS_received</th>\n",
       "      <th>No-show</th>\n",
       "      <th>due-days</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>F</td>\n",
       "      <td>62</td>\n",
       "      <td>JARDIM DA PENHA</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>M</td>\n",
       "      <td>56</td>\n",
       "      <td>JARDIM DA PENHA</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>F</td>\n",
       "      <td>62</td>\n",
       "      <td>MATA DA PRAIA</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>F</td>\n",
       "      <td>8</td>\n",
       "      <td>PONTAL DE CAMBURI</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>F</td>\n",
       "      <td>56</td>\n",
       "      <td>JARDIM DA PENHA</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Gender  Age      Neighbourhood  Scholarship  Hipertension  Diabetes  \\\n",
       "0      F   62    JARDIM DA PENHA        False          True     False   \n",
       "1      M   56    JARDIM DA PENHA        False         False     False   \n",
       "2      F   62      MATA DA PRAIA        False         False     False   \n",
       "3      F    8  PONTAL DE CAMBURI        False         False     False   \n",
       "4      F   56    JARDIM DA PENHA        False          True      True   \n",
       "\n",
       "   Alcoholism  Handcap  SMS_received  No-show  due-days  \n",
       "0       False    False         False     True         0  \n",
       "1       False    False         False     True         0  \n",
       "2       False    False         False     True         0  \n",
       "3       False    False         False     True         0  \n",
       "4       False    False         False     True         0  "
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 110521 entries, 0 to 110526\n",
      "Data columns (total 11 columns):\n",
      " #   Column         Non-Null Count   Dtype   \n",
      "---  ------         --------------   -----   \n",
      " 0   Gender         110521 non-null  category\n",
      " 1   Age            110521 non-null  int64   \n",
      " 2   Neighbourhood  110521 non-null  object  \n",
      " 3   Scholarship    110521 non-null  bool    \n",
      " 4   Hipertension   110521 non-null  bool    \n",
      " 5   Diabetes       110521 non-null  bool    \n",
      " 6   Alcoholism     110521 non-null  bool    \n",
      " 7   Handcap        110521 non-null  bool    \n",
      " 8   SMS_received   110521 non-null  bool    \n",
      " 9   No-show        110521 non-null  bool    \n",
      " 10  due-days       110521 non-null  int64   \n",
      "dtypes: bool(7), category(1), int64(2), object(1)\n",
      "memory usage: 4.2+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We endded up with a datafram of 110521 rows and 11 columns, and everything looks tidy and clean. We'd proceed in visualizing it to extract meaningful insights from it.\n",
    "___"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data Visualization and EDA\n",
    "Now that our data is clean, we'd perform some EDA on it in order to extract useful insights from it."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "# setting seaborn configurations\n",
    "sns.set_style(\"whitegrid\") "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### **How often do men go to hospitals compared to women? Which of them is more likely to show up?**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYwAAAESCAYAAADuVeJ5AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAAdd0lEQVR4nO3df1RUdf7H8ddlUCwGMnY1DwkeNS0syZD1xx60rdWozCwzwVzdVlvL/JEedSEVEDV/pIuZllrbbqtb6aq02ebWbpa5gqG54a8mK9dUAn8U/hrSUZj7/cPjfGNT/CAMA/h8nNM5zee+72fety7z4n5m7mDZtm0LAIBLCAp0AwCAuoHAAAAYITAAAEYIDACAEQIDAGCEwAAAGCEwUOesXLlSDz/8sO655x716NFDv/nNb7Rt27ZqfY6pU6dqwYIFVZojNTVV3bp1U58+fdSnTx/de++9GjdunI4cOSJJOnTokJKTkyuc48CBAxo1atQFt/1w/wULFmjq1KmV7nHy5MnauXOnJGnSpEnKzc2t9By4cgQHugGgMrKysrRlyxY999xzuv766yVJmzZt0uOPP67s7GxFRkYGuMPyHn30UQ0dOlSSZNu2lixZoscee0zZ2dm67rrrtHz58gr3Lyws1N69ey+4zWT/S8nNzVVSUpIk6ZlnnqnSXKj/uMJAnfHtt9/qz3/+s+bPn+8LC0nq2rWrUlNTderUKUnnfvMeMWKE+vbtq969e2vx4sWSpIKCAvXo0UPTpk1Tv3791LNnT61du1aS5Ha79dRTTykxMVGDBg3Sf//7X9/8Fc13++23a8iQIUpMTNThw4cr7N+yLD3xxBM6ffq0cnJyVFBQoNtuu02StGfPHiUnJ6tv37568MEH9dprr6msrEyTJ0/W/v37NXTo0B8936effurb//wcAwcO1H333acJEybI7XZLku68807t2LHDV3f+8bx583T48GGNHz9e27Zt06BBg/Tuu+9Kkt5//3098MAD6t27twYMGKDt27dLOnclk5qaqqFDh+ruu+/WI488okOHDl3G/03URQQG6oz8/Hy1bt1aTZs2/dG2Bx54QK1bt5YkTZgwQQ899JCys7O1atUq5ebm+oLhwIEDSkhI0KpVqzR+/HjNmTNHkvT888+rUaNGevfddzV//vxyv9VXNN/Bgwf15JNP6r333rtgXxdy44036osvvig39sorr+jOO+9Udna2XnrpJX3yySeyLEvTp09XdHS0XnnllR89X5MmTcrNsX//fi1YsEBvv/22bNvWokWLKuxj7Nixatq0qebOnatbb73VN75nzx5lZGT45ho9erSefPJJXwB98sknmj9/vt59912Fh4drxYoVRseNuo8lKdQZ//stNm63WwMHDpQkff/997rnnnv0xBNPaMuWLTp+/Ljmz5/v2/b5558rNjZWDRo00O233y5JateunY4dOybp3LLWxIkTZVmWIiIi1LNnT9++Fc0XHBysDh06VOo4LMvSVVddVW6sZ8+eSklJ0fbt29W1a1dNnjxZQUE//n2uoufr2bOnIiIiJEkPPfSQnn322Ur1dd7HH3+sLl26KCoqStK5K7iIiAjfex2dOnWS0+mUdO6/4fHjxy/reVD3EBioM2JjY7V3714dPXpU1157rZxOp9566y1J55ZKjh49Kq/XK9u2tXz5ct+LcnFxsUJCQnT06FE1aNDA90JsWVa5+X8YSA6HQ5IuOV/Dhg0VHGz+Y2Tbtnbt2qVf/epX5cbvuOMOvffee8rNzdWmTZv0wgsvXPD9iYqe73zP55/nh3U/PLYzZ85csscLjZWWlkqSGjVq5Bu3LOuC9aifWJJCnXHddddp8ODBeuqpp1RYWOgbLyws1H/+8x8FBQXJ6XSqQ4cO+tOf/iRJOnHihAYMGKB169ZVOHe3bt20atUqeb1eHT9+3Fd/ufNdSFlZmV544QVde+21+tnPflZu27hx47R27Vr16tVLGRkZcjqdKioqksPh0NmzZ43m/+CDD3T8+HGVlZVpxYoV6t69uySVuzrIz8/3fUpLOhcy54PgvC5duignJ0cHDhyQdO7qq6ioqNyyFa5MXGGgThk7dqzWrFmj8ePH6/vvv1dpaakaNmyoe++917c8NXfuXE2bNk29e/fWmTNndN999+n+++9XQUHBRecdNWqUMjIydM899ygiIkJt27b1bbuc+c579dVXtWbNGlmWpbKyMrVv314vvfTSj+qefPJJTZo0SStWrJDD4VCPHj3UqVMnnThxQg6HQ/369dO8efMqfK7WrVvr8ccf14kTJ9SxY0cNGzZMkjR+/HhNmTJFK1as0M0336ybb77Zt0+PHj00duxYTZ8+3Td2ww03KCMjQyNHjlRZWZkaNWqkxYsXKyws7JLHi/rN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BuLQcQUjMIDLUFBQoOjoaN/j4cOHy+126/Dhw2rWrJmGDBmi22+/XZs2bdLcuXM1duxYff3113rllVd01VVXqUePHjpy5IgWL16s++67T/3799fatWv1xhtvyLZtpaWl6fXXX9dPfvITPffcc3rzzTcVHBys8PBwLVq0KIBHjisZgQFchmbNmmnnzp2+x+dfxPv376/8/HwtWbJEf/jDH2TbtoKDz/2YRUdHy+l0SpKaNGkij8ejr7/+Wv3795ckxcXF6Y033lBxcbEOHz6sMWPGSJJOnz6tn//852rRogVff4GAIjCAy/DLX/5SL7/8svLz89WhQwdJ0r59+3Tw4EHFxsZq7NixiouL0549e7RlyxZJkmVZP5qndevW+vTTT3XTTTdpx44dkqRrr71WzZo104svvqiwsDCtW7dOV199tYqKihQUxOdUEDgEBnAZQkNDtWjRIv3+97/X3LlzVVpaKofDoaefflq33HKLpkyZIo/Ho9OnT2vSpEkXnWf48OGaMGGC1q5dq+bNm0uSgoKCNGnSJA0bNky2bSs0NFTPPvusioqKaurwgAviywcBAEa4vgUAGCEwAABGCAwAgBECAwBghMAAABghMAAARggMAICR/wMCOOZUTNyCxgAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# viewing count plot of gender distribution in our dataset\n",
    "sns.countplot(x = 'Gender', data = df)\n",
    "plt.title(\"Gender Distribution\")\n",
    "plt.show()"
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
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# viewing count plot of gender distribution against age in our dataset\n",
    "sns.boxplot(x = 'Gender', y = 'Age', data = df)\n",
    "plt.title(\"Gender Distribution against Age\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- we can notice that nearly half of our dataset conists of women with wider age destribution and some outliers, all of which achiees a rate higher than men."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "True     88207\n",
       "False    22314\n",
       "Name: No-show, dtype: int64"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['No-show'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- it is obvious that 79.8% of our patients did show up on their appointments and only 20.1% of them did not.\n",
    "\n",
    "lets dive deeper to see if this is related to gender."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYwAAAESCAYAAADuVeJ5AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAArI0lEQVR4nO3de3zO9f/H8ce1XTPaAVMqmX1RwhfNLMLMMaNIB4w0uUn6qojCHOdUDsUqlFB9v31XUU7f+qb6YdRyGpFDa1QUZnP40Wab7drhev/+8HX92tfGBzvheb/duuX6XO/r83l9Pnvv89z7c72vz2UzxhhEREQuwa2sCxARkWuDAkNERCxRYIiIiCUKDBERsUSBISIiligwRETEEgXGNeLrr78mIiKiWNc5depU5s2bd8WvnzBhAj/++GOhz40fP57NmzeTlJRE06ZNL3vd33zzDW+++SYAsbGxvPzyy1dc5+VasmQJixYtAmDZsmV89NFHAMybN4+pU6eWWh3FJT09nf79+xf63J49e4iKiirlii7u7rvv5vTp02VdRgFX2o+vN/ayLkCuXZs3byY8PLzQ51555RXg3C/aldi7dy9paWkAdOzYkY4dO15ZkVegb9++rn/v2LGDu+66q9S2XRLS0tLYu3dvoc/9+uuvHD9+vJQrkmuVAqMYLVq0iOXLl+Pl5UVwcDCxsbGsX7+enJwcZs+ezfbt28nPz6dhw4ZMmDABb29vOnTowCOPPMKWLVtISUmha9eujB49GoA333yTf//731SpUoWAgADXdi61viZNmrB//35efPFF7r//ftfrMjIyGD9+PPv27aN69eq4u7vTrFkzAI4fP87UqVNJSUkhNzeXBx98kL/97W/k5eUxbdo0du7ciYeHBzVr1mTGjBksWrSIEydOMHLkSF599VVmz55N5cqVOXjwIH379mXNmjX069ePRo0a4XQ6GT9+PAkJCdjtdiZMmEBgYCDz5s3jjz/+cP2Fe/5xjx49WLp0Kfn5+fj4+BAQEMD//M//sHDhQo4dO8bkyZM5evQoxhgefvhhBg0aRFJSEgMGDKBt27bs3r2btLQ0RowYwQMPPFDgZ/Tcc8/Rrl07evXqxa5duwgPD2fdunX4+/uzYMEC0tPTqVSpEn/88QctW7Zk/fr1bNq0iYoVKwJw8OBBIiIiOHnyJDfffDPR0dFUr169wDbmzZvHrl27OHHiBHfffTezZ89mwYIFrFmzBqfTyR133MGkSZO49dZbWbNmDQsWLMBms+Hu7s7o0aO59957iYiIoG7duvz444+uYzJs2DAAdu7cyezZs8nKysJmszF06FDat28PwMKFC1m1ahV2u52AgABmzpzJ2LFjyc7OpkePHqxcuRJ3d3cAUlJSmDt3Lunp6YwdO5YZM2bwySefEBMTg5ubGzfffDMTJ06kdu3aBfYvPj6e119/HX9/f3755RdycnKIiorivvvuIz09nSlTprBv3z5sNhtt2rThxRdfxG6/8FQzd+5c1q5di4eHB1WrVmXGjBmuYzlv3jx2795NamoqTz31FP369QPgrbfeYvXq1bi7u1O7dm0mTpzI7t27ee+991iyZAkAXbp0oWvXrrzwwgscO3aMnj17EhcXh5vb/19QOXnyJJMmTeLgwYO4ubnRp08f+vfvX2T/+u+fb2H9NioqioiICAIDA9m5cycpKSk0a9aMWbNmkZyczJNPPsl9993Hrl27yMvLY/To0XzyySccPHiQRo0aER0dTXJysqV+XGaMFIu4uDgTFhZm0tLSjNPpNGPHjjXt27c3xhgzb948M3PmTON0Oo0xxsyZM8dMmjTJGGNM+/btzcyZM40xxhw7dsw0btzYHD582Kxdu9Y88MADJj093eTm5prBgwebJ554wtL65s+fX2iNr7zyihk9erRxOp3m1KlTJjQ01MydO9cYY0xERISJjY01xhiTnZ1tIiIizOrVq8327dtNly5dXNt69dVXzY4dO1zb2rNnjzHGmCeeeMKMHTvWta0nnnjCfPXVV+bIkSOmXr16ZvXq1a7j1LZtW+NwOMzcuXPNlClTXK/58+M//3vFihVm8ODBxhhj+vXrZ95//31jjDFnzpwx3bt3N1988YVrO+vXrzfGGPP111+bdu3aXXAMVq1aZYYOHeraRuvWrc3SpUuNMcY89thjZvfu3QW2HRkZad59911X+w4dOphTp04ZY4wZMmRIocd67ty5JiwszOTm5rq2OXz4cNfjpUuXmkGDBhljjOnYsaP54YcfjDHGfPfdd2bevHmu4/f000+bnJwck5aWZsLCwsz69etNamqq6dy5szly5Igx5lyfCQ0NNUePHjXr1q0znTt3NqmpqcYYY6ZPn27efvttc+TIERMYGHhBnf99bDdv3mw6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y5YtiY+PB87dbyY4OJigoCA2btyI0+kkOTnZ9WGhhg0bWm4rIiKlo0RGGO3bt2f79u307NkTYwxRUVHUrFnTdQuLOnXqEBYWhru7O8HBwYSHh+N0Ol23C46MjLTcVkRESsd1/TmMxMREXZISEctyc3NJSkoiOzu7rEspFRUrVnTdSffPijp36pPeIiL/kZSUhI+PD3/5y1/K3Z1ii5sxhlOnTpGUlHTBl2QVRXerFRH5j+zsbKpVq3bdhwWcu3V6tWrVLms0pcAQEfmTGyEszrvcfVVgiIiIJQoMEZErEB8fT7NmzUhJSXEtmz17tusbE69E69ati6O0EqPAEMtMnuPSjW4AOg5yXoUKFRg7duwN8wVKmiUlltnsnhye2risyyhztaL2lnUJUk7cd999OJ1OPvroI5544gnX8vfff5/Vq1djt9sJDg5m1KhRBV7ncDh44YUXyMjIICsrixEjRhASEkJOTg4vvfQSycnJVKlShblz55KVlcWoUaPIyMggPz+fF154gczMTDZv3kxUVBSLFi1i586dvPPOO3z++eckJycX+i2HxUGBISJyFSZPnkyvXr1o06YNcO62RV999RVLly7FbrczdOhQNmzYQPv27V2vOXz4MKmpqbz77rucOnWK33//HYCzZ88yYsQIatasSUREBImJiXz11Ve0atWKJ598kuPHj9O3b1++/PJL3nzzTQC2b9/OqVOnyMvLY/369QwdOrTE9lWBISJyFapWrcq4ceOIjIwkKCgIh8PBPffc4/owXHBwML/88gvr1q3j8OHDVK1alblz5xIeHs6LL75IXl4eERERAFSuXJmaNWsCcPPNN5OVlcWBAwfo3r07ALfeeive3t5kZGRQu3Zt9uzZg91u55577mH79u2kpKRQt27dEttXBYaIyFXq0KEDa9euZdWqVTz77LPs2bOHvLw83N3d2b59Ow8//DCDBw92td+/fz+ZmZksWrSIEydO0KdPH9q3b1/oNNe6devy/fff07BhQ44fP86ZM2eoUqUKnTp14rXXXqNjx474+/vz+uuv06pVqxLdT73pLSJSDMaPH0/FihXx8vKia9eu9O3bl549e3LHHXfQqVOnAm3/8pe/sG3bNvr168cLL7zAsGHDilzvM888w9atW+nXrx/PPvssU6dOxW630759e3744QdCQkJo0aIFP/30E507dy7RfdS9pOSy6E1vvel9PbsRzxmF7XNRx0EjDBERsUSBISIiligwRETEEgWGiIhYosAQERFLFBgiIkVw5OaX6/WVNn1wT0SkCJ4e7jQb9c9iW9+O1/pfsk1SUhIPPfQQf/3rX13LWrRowfPPP39B2zFjxvDAAw8QGhpabDVejAJDRKScufPOO4mJiSnrMi6gwBARKefy8/OJiori2LFjnDhxgg4dOjBixAjX87/99htjx47FbrfjdDqZM2cOt99+O3PmzOH777/H6XQyYMAAunbtelV1KDBERMqZX3/91XVDQoDhw4cTGBhIr169cDgchIaGFgiMzZs306RJE0aNGsX3339Peno6P//8M0lJSSxZsgSHw0Hv3r1p3bo1vr6+V1yXAkNEpJz570tSGRkZfPbZZ2zduhVvb29ycnIKtO/ZsyeLFy9m0KBB+Pj4MGLECH7++WcSEhJcwZOXl8fRo0evKjA0S0pEpJxbuXIlPj4+zJkzh4EDB5KdnV3gW/5iY2Np1qwZH3zwAV26dOHdd9+lTp06tGjRgpiYGD744AO6du2Kv7//VdWhEYaISBEcufmWZjZdzvo8Pdwv+3UtW7bkpZdeYteuXVSoUIGAgABOnDjher5Ro0ZERkayYMECnE4nY8eOpWHDhmzbto3HH3+cs2fP0qlTJ7y9va+qft2tVi6L7laru9Vez27Ec4buVisiIsVOgSEiIpaU2HsYjzzyiOt6Wc2aNQkPD+eVV17B3d2dkJAQnn/+eZxOJ5MnT2b//v1UqFCBl19+mYCAAHbt2mW5rYiIlI4SCQyHw4ExpsC0sB49ejBv3jz8/f0ZPHgwP/30E0lJSeTk5PDJJ5+wa9cuZs6cyYIFC5g0aZLltiIiUjpKJDD27dtHVlYWAwcOJC8vj6FDh5KTk0OtWrUACAkJYfPmzZw8eZI2bdoAEBgYyI8//khGRobltiIiUnpKJDAqVqzIU089Ra9evfj99995+umnC3xYxMvLiyNHjpCRkVFgmpe7u/sFyy7WNi8vD7u96F1wOBwkJiYW897duG602SMXo351fcrNzSUrK8v12NNuw82jYrGt35mbjSOvfE1Mzc3NtdyfSyQwateuTUBAADabjdq1a+Pj40Nqaqrr+czMTHx9fcnOziYzM9O13Ol04u3tXWDZxdpeLCwAPD09dZKTEqF+dX1KTEykUqVKBZYV51TyWlF7qeRR9PMzZ84kISGBkydPkp2djb+/P1WrVmXu3LnFVsN/8/DwKHRabWFKZJbU8uXLmTlzJgDHjx8nKyuLm266icOHD2OMYePGjQQHBxMUFERcXBwAu3btol69enh7e+Ph4WGprYjI9WTMmDHExMQwePBgunXrRkxMTImGxeUqkRFGz549GTt2LH379sVmszF9+nTc3NwYOXIk+fn5hISEcM8999C4cWM2bdpEnz59MMYwffp0AKZMmWK5rYjI9WzMmDGkpqaSmprKU089xZdffsnrr78OQOvWrdm0aRMpKSlMnDgRh8OBp6cn06ZN4/bbby/2WkokMCpUqMCcOXMuWP7pp58WeOzm5sbUqVMvaBcYGGi5rYjI9e6+++5jwIABxMfHF/r8rFmziIiIoG3btmzZsoXZs2cXeg6+WrqXlIhIOVe7du1Cl5+/s9PPP//MwoULeffddzHGXPL93SulwBARKedsNhtwbiLPyZMnATh69ChpaWkA1KlTh4EDBxIUFMSBAwfYvn17idShwBARKYLJcxTrzSZNngOb3fOKX9+oUSN8fHzo1asXdevWpWbNmgBERkYyefJkHA4H2dnZjB8/vrhKLkCBISJShKs5uV/N+h599FHXv8/POAWw2+2F3uHC39+f99577+oLvATdfFBERCxRYIiIiCUKDBGRP7mOv1PuApe7rwoMEZH/qFixIqdOnbohQsMYw6lTp6hY0fq9svSmt4jIf9SsWZOkpCTX1NXrXcWKFV0zraxQYIiI/IeHh0eRH5ITXZISERGLFBgiImKJAkNERCxRYIiIiCUKDBERsUSBISIiligwRETEEgWGiIhYosAQERFLFBgiImKJAkNERCxRYIiIiCUKDBERsUSBISIiligwRETEEgWGiIhYosAQERFLFBgiImJJiQXGqVOnaNu2LQcOHODQoUP07duXxx9/nEmTJuF0OgGYP38+PXv2pE+fPuzZswfgstqKiEjpKZHAyM3NJSoqiooVKwIwY8YMhg8fzscff4wxhtjYWBISEti2bRvLli0jOjqaKVOmXHZbEREpPSUSGLNmzaJPnz5Ur14dgISEBJo3bw5AaGgomzdvZseOHYSEhGCz2ahRowb5+fmcPn36stqKiEjpsRf3CleuXImfnx9t2rRh0aJFABhjsNlsAHh5eZGenk5GRgZVqlRxve78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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# showing the gender destribution with respect to the no-show colunmn\n",
    "sns.countplot(x = 'Gender', data = df, hue = 'No-show')\n",
    "plt.title('gender destribution with respect to no-show colunmn')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- from the above chart, we can come up with a conclusion that women do show up on their appointments more often than men do, but this may b affected by the percentage of women on this dataset."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "___\n",
    "### **Does recieving an SMS as a reminder affect whether or not a patient may show up? is it correlated with number of days before the appointment?**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYwAAAESCAYAAADuVeJ5AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAAjLklEQVR4nO3df1xUdb7H8dcAMiUDcedu2yOuPwINw1XiIquVSEu5mmtquRKIS3pN2+2qPazVQEsUf6FrYpmbmt12H2prhtJW9nOjNRUVjbusShOm10hEzZtpgDoI871/9Gh2LdHDhQGF9/Mv5sznnPM5w+G8+Z6Zc8ZmjDGIiIhchl9LNyAiIlcHBYaIiFiiwBAREUsUGCIiYokCQ0RELFFgiIiIJQoM8bni4mLS0tIYMmQI9957L+PGjeOzzz4DoLy8nG7dujFq1KgfzDdt2jS6devGyZMnL7uc5pSfn8/cuXObZFknT56kW7duTVZXnz179pCZmXnR5/bu3cujjz4KQEZGBv/1X//V4OWPHTvW+3saP348Bw4c+H/3KleugJZuQFq3mpoafv3rX/PSSy/xk5/8BIDXX3+d8ePHk5+fD4Ddbufzzz/nyJEj/Nu//RsAZ86coaioyPJy/P39m22b7r77bu6+++5mW19TOHDgAMePH7/ocz179mTp0qWNWn5BQYH351WrVjVqWXLlUmCIT509e5bKykrOnDnjnTZ06FAcDgd1dXUA+Pv7M2jQIN58801+85vfAPD+++9z991389JLL1lazvcD46677iI6OprS0lIef/xxoqOjmT17NkePHuX8+fMMHjzYu66//vWvPPPMM3g8Htq3b09WVha33HIL//3f/83TTz/N2bNnsdlsTJo0icTERPLy8njvvffIyMggJSWFrVu3EhgYSF1dHYmJibz00kvccMMNzJs3j/3793P+/Hluv/12nnjiCQICAnj//fdZsmQJ1157LT169Kj3tbtUXW5uLuvWrcPj8RAaGsqMGTPo0qULH3/8MQsWLMDj8QDw61//mujoaJYuXUplZSXTpk3jvvvuY968ebRv354zZ84wdepUFi5cyKZNmwAoKirivffeo6qqir59+5Kenk5AQADdunVjx44dOJ1OAO/jRYsWATB69GheeOEFRo0axbPPPkvPnj1Zv349a9aswc/Pjx/96EfMmDGD8PBwMjIy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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# viewing count plot of recieving SMS distribution in our dataset\n",
    "sns.countplot(x = 'SMS_received', data = df)\n",
    "plt.title(\"SMS received destribution\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "False    75039\n",
       "True     35482\n",
       "Name: SMS_received, dtype: int64"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['SMS_received'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- we can see that 67.8% of our patients did not reciee any SMS reminder of their appointments, cound this be affecting their showin up?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYwAAAESCAYAAADuVeJ5AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAAtuklEQVR4nO3deUAVdb/H8fdhT8CFFm8+bqBZmqkh1xU1jDRL1GsYqFFm2qqFqSGpuOZSLrlbeu32mKWZlu2LWHGFXMtUIk1yQ0h5XAH1sJzf/cPreSRRx2RLPq+/ODO/mfOd8ed8zuw2Y4xBRETkClzKugAREfl7UGCIiIglCgwREbFEgSEiIpYoMERExBIFhoiIWKLAKAXbtm0jKiqKsLAwunbtyoABA/jtt98ASEtL4/bbb6dv374XTRcbG8vtt9/OsWPHrjifK7n77rtJS0v7S/UfPHiQwYMHFznu8OHDREZGAjBnzhzGjx9/1fMfNWoUO3fuBGDkyJEkJSX9pTr/ioEDB7Jnzx4A+vfv71zXHTt2ZMeOHaVWR3H57rvvmDVrVpHj5s6dy9q1a0u5oktbvXo1Tz31VFmXcZG/2o8rAreyLuB6l5uby1NPPcWSJUu48847AVizZg0DBw4kPj4eAE9PT/bt28ehQ4f4xz/+AcDp06fZunWr5fm4urqW2DKkp6ezd+/eIsdVr16d5cuXX9P8k5KSiIiIAOCVV165pnldrUWLFjn/TkxMLNXvLgk7duzg5MmTRY7buHEj9evXL+WK5HqiwChhZ86cISsri9OnTzuHdevWDR8fHwoKCgBwdXWlS5cufPLJJzz99NMAfP3119x7770sWbLE0nz+HBhbtmxhwoQJ2Gw27rrrLhwOh3PcunXrWLBgAXl5eXh5eRETE8Pdd99NamoqI0eOJDc3F2MM4eHhREZGMmrUKA4fPswTTzzBuHHj6Nu3L/Xq1ePQoUNMmTKF/v3789NPPwGQmppK3759OXnyJA0bNmTMmDH4+PjQsWNHZs2axV133QXg/Lx27VqOHDnCsGHDePXVV5k2bRp9+/bl/vvvZ+3atcydO5eCggJ8fHyIjY2lSZMmzJkzh0OHDpGZmcmhQ4fw8/Nj5syZVK9e3bmMx48fJyQkhKSkJCpVqkRcXBypqaksW7YMgE6dOjF//nyefPJJZs2axbvvvgvAY489xptvvgnAihUrGDNmDMeOHaN79+4MGTLkon/fjh070qRJE3bt2sWLL75IkyZNGD9+PBkZGeTl5fHggw/y9NNPk5+fz4QJE/jxxx9xd3enZs2aTJ48mePHjxMVFUWLFi349ddfMcYQFxdHUFAQAAsWLODrr7/G4XDwj3/8gzFjxlC9enUyMzMZM2YMv//+Oy4uLkRGRtK0aVOWL19OQUEBvr6+hepdtmwZO3fu5NVXX8XV1ZVWrVoxbtw4fv31V2w2G+3atePFF1/Eza3wJmHEiBH4+Piwa9cu/vjjDwICApgxYwbe3t5s2bKFV199lTNnzuDu7k50dDTt27e/aB1lZmYSExPD8ePHAejQoQPR0dHOcU8++SQZGRm4uroyffp06tWrxx9//MHYsWM5dOgQxhh69OjBgAEDeO6557jnnnvo1asX27ZtIyIigrVr11KrVi0WLFhAVlYWL730UqHv//bbb3n99ddxOBxUqlSJcePGcccdd1yyf/3537eoflutWjX69etHhw4d+Pnnnzl58iRDhgzhgQceYM6cORw4cICDBw9y5MgRmjRpQtu2bfnoo49IS0tj+PDhdO3a1VI/LneMlLglS5aYJk2amI4dO5phw4aZlStXmtOnTxtjjDl48KBp1qyZ2bFjh+nSpYtzmscee8zs2rXLNGjQwBw9evSK87mQ3W43bdq0MUlJScYYYz755BPToEEDc/DgQbN3717TtWtXc+zYMWOMMbt37zZt27Y1OTk5JjY21rzxxhvGGGOOHDlioqOjTUFBgdmwYYN58MEHnfU2aNDAbN68uVD9xhgze/Zsc88995ijR48ah8Nhhg4dal599VVjjDEhISFm+/btzhov/Hzh34888oj54osvzJ49e0ybNm3MgQMHjDHGJCUlmbZt25qsrCwze/Zsc++995qsrCxjjDFPPfWUmTVr1kXrISoqyqxbt84YY0ynTp1MmzZtTHZ2tvntt9+c6/rC775wXYeEhJjx48c710Xjxo1Nenr6Rd8REhJi5s6dW+g74+PjjTHGnD171kRFRZnPPvvMbN682dx///3G4XAYY4x59dVXzdatW53r8+OPPzbGGPPdd9+Ztm3bmtzcXPPhhx+a6Ohok5eXZ4wxZvny5WbAgAHGGGOee+45M3XqVGOMMadOnTIPPvig2bdvn5k9e7YZN27cRXVeuG6NMeall14yEyZM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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# showing the sms destribution with respect to the no-show colunmn\n",
    "sns.countplot(x = 'SMS_received', data = df, hue = 'No-show')\n",
    "plt.title('SMS destribution with respect to no-show colunmn')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- we can see that our previous deduction was not quiet correct, as the vast majority of our patients did not recieve any SMS reminder and yet they showed up on their appointments."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAXwAAAESCAYAAAD+GW7gAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAAgBUlEQVR4nO3deVjU5f7/8ecAggKS0kLmdlwT81iHzA20NIw0TUsMzEhzq+MGbrkGmuWSWint6ulcqWmetKOVZWqdOG6kp6wTooi5hqFpIlsDMvfvD3/ON4+KUzEM8Xk9rsvrYuaz3O/PPeNr7vlsYzPGGEREpNLz8nQBIiJSPhT4IiIWocAXEbEIBb6IiEUo8EVELEKBLyJiEQp88aibb76Z06dPe7qMX2XIkCFkZmb+pmW/+eYbEhMTXZr3L3/5C8eOHftN7biie/fupKamum39UvH4eLoAkT+aRYsW/eZlMzMzyc7OLsNqRFynwJdSpaam8sILL1C3bl32799PUVERiYmJtG3bltzcXKZPn87evXux2Wx06NCBMWPG4ONz6dtq4cKFbNy4kSpVqlCzZk1mzZrFDTfcAEBycjJff/01Z86cYdCgQfTr1w+Al19+mQ8//BBvb28aNGjAU089xddff82SJUtYsWIFAPfeey9du3YlPj6eH374gejoaFJSUvDy+r8vr7t372bu3LkUFRVx8uRJ2rdvz8yZMwFYs2YNb7zxBlWrVqVt27a89dZb7Nmzhx9//JHExEROnTrFyZMnqV27Ni+++CLXXnstnTt3ZsGCBRQUFFyxb3bt2sXs2bNxOBwAPP7447Rs2ZKFCxeSm5vLpEmTmDVr1kV9tGvXLmbMmIHNZuPPf/6zc9nU1FRmzJjBBx98cNnHr776Kp988gkOh4PatWuTlJRESEjIJa9BZmYmkydPprCwkIYNG1JQUOCc9tprr7Fp0ybsdjuFhYVMmDCByMhI7r33Xp566ikiIiIAmDp1Kk2aNCEiIoIpU6ZQVFSEMYbo6Gjn6yYVmBEpxY4dO0xoaKjZs2ePMcaYJUuWmH79+hljjHnyySfNjBkzjMPhMHa73QwcONC8/vrrl6wjKyvLhIWFGbvd7lzHxo0bjTHGNG3a1CxZssQYY0xaWppp0aKFKSoqMu+++66JiYkx+fn5xhhjFi5caAYOHGgKCwtNWFiYycnJMUePHjXh4eEmJibGGGPMsmXLTFJS0iXtjx492uzYscMYY0xeXp5p06aN+e9//2v2799v2rVrZ44fP26MMSY5Odk0bdrUGGPM3//+d+e2OBwOM3jwYGednTp1Mt98802pffPoo4+aDz74wBhjTHp6upk2bZoxxpjVq1eboUOHXlKj3W437du3N9u2bTPGGPP++++bpk2bmqNHj5odO3aY++6776LX5MLj9957zyQkJJji4mJjjDErV640gwcPvtxLaXr27GlWrVpljDFm165d5uabbzY7duwwx44dM3FxcaawsNAYY8wHH3xgunfvbowx5s033zSjRo0yxhiTm5tr2rZta3JycsykSZOc/XPixAmTkJBgSkpKLtuuVBzahy9XddNNNxEaGgpA8+bNycnJASAlJYVHHnkEm82Gr68vsbGxpKSkXLJ8SEgIzZo144EHHmDOnDmEhoYSGRnpnN69e3cAQkNDKSoqIi8vj5SUFB588EH8/f0BePTRR9mxYwdeXl60b9+erVu38u9//5uYmBiOHTtGbm4un376KVFRUZe0P3v2bHJzc3nttdeYPn06P//8MwUFBWzZsoXw8HBuvPFGAB555BHnMv379ycsLIw333yTadOmsX///otGxFfrm65du/L0008zduxY0tLSGDNmTKl9nJGRgY+PD+3atXP2SUBAQKnLAHz22Wd8/fXX9O7dm549e7Js2TIOHjx4yXw//fQT+/bto1evXgDcfvvtNGnSBIDatWszZ84c3n//febNm8fKlSvJz88H4MEHH2Tbtm2cPn2adevWcddddxEUFESXLl1YvHgxI0aM4JNPPmHq1KkXfauSikmvkFxV1apVnX/bbDbM/7/90oVdDhc4HA7OnTvH5s2b6dmzJz179mTIkCF4eXmxbNkyZs2aRY0aNZg5cybPPPOMc7kLu4BsNhsAxhhnG/+7boAuXbqQkpLCli1biIiI4I477mDTpk1kZGRwxx13XFJ/v379+Pzzz2nYsCHDhw8nJCQEYwze3t4XtePt7e38e+7cuSxYsICaNWsSExNDeHj4JTWV1jexsbGsW7eO8PBwtmzZwv33309ubu4V+/iXy16uX345rbi4+KJ+GTx4MGvXrmXt2rWsXr2aFStWkJ2d7XwNevbsSVFRkbNv/3f9aWlpxMbGkpeXR3h4OIMHD3bOExQUxL333su6detYvXo1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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# viewing the correlation between no-show and due-days without outliers\n",
    "sns.boxplot(x = 'No-show', y = 'due-days', data = df, showfliers = False)\n",
    "plt.title('no-show against due-days')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- it is clear that there is a correlation between number od due days and whether a patient shows up or not.\n",
    "- patient with appointments from 0 to 30 days tend to show up more regularly, while patients with higher number of days tend to not show up. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# viewing the correlation between no-show and due-days without outliers with respect to gender\n",
    "sns.boxplot(x = 'No-show', y = 'due-days', data = df, hue = 'Gender', showfliers = False)\n",
    "plt.title('no-show against due-days')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- gender does not affect number of due days and showing up at an appointment that much."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "___\n",
    "### **Does having a scholarship affects showing up on a hospital appointment? What are the age groups affected by this?**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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Li4OgLlz53q4EhHxFJ0TEBExmEJARMRgCgEREYMpBEREDKYQEBExmEJARMRgCgEREYMpBEREDKYQEBExmEJARMRgCgEREYMpBEREDKYQEBExmEJARMRgCgEREYMpBEREDKYQEBExmEJARMRgCgEREYMpBEREDKYQEBExWI2FwO7du7FarQDk5+czaNAgYmJiSExMxOFwAPDnP/+Z/v37M3DgQPbs2VNTpYiIyGXUSAgsWrSIKVOmYLfbAZg1axZjxoxhxYoVOJ1O0tPTyc7OZuvWrXz88cckJyfz8ssv10QpIiLyM2okBJo0aUJKSorrdXZ2Nh07dgSge/fubNmyhe3bt9O1a1csFgs333wzFRUVFBUV1UQ5IiJyGT410WhUVBRHjx51vXY6nVgsFgACAwMpLi6mpKSEa665xrXM+emhoaE/27bdbicnJ6cmyjaOzWYD0PYUMViNhMD/8vL674CjtLSUkJAQgoKCKC0trTQ9ODi4yrb8/PyIiIiokTpNExAQAKDtKWKAy33Zc8vVQa1atSIrKwuAjIwM2rdvT7t27di8eTMOh4OCggIcDkeVowAREbmy3DISmDBhAlOnTiU5OZmwsDCioqLw9vamffv2DBgwAIfDQUJCgjtKERGRC1icTqfT00X8Gjk5OTp8cYXExcUBMHfuXA9XIiI17XKfnbpZTETEYAoBERGDKQRERAymEBARMZhCQETEYAoBERGDKQRERAymEBARMZhCQETEYAoBERGDKQRERAymEBARMZhCQETEYAoBERGDKQRERAymEBARMZhCQETEYAoBERGDKQRERAymEBCRWqewsJDRo0dTWFjo6VLqPIWAiNQ6S5YsYe/evSxdutTTpdR5CgERqVUKCwtZv349TqeT9evXazRQwxQCIlKrLFmyBIfDAUBFRYVGAzVMISAitcoXX3xBeXk5AOXl5WzcuNHDFdVtCgERqVV69uyJj48PAD4+PvTq1cvDFdVtPp4uQERqlw0bNpCWluax/svKylwjgYqKCg4ePEhcXJzH6omOjiYqKspj/dc0jQREpFapV6+eayQQGhpKvXr1PFxR3ea2kUBZWRnx8fEcO3YMLy8vZsyYgY+PD/Hx8VgsFlq0aEFiYiJeXsolEU+Kiory+DffP/3pT+Tn57Nw4UKuu+46j9ZS17ktBL788kvKy8v58MMPyczM5K23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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plotting having a scholarship against age\n",
    "sns.boxplot(x = 'Scholarship', y = 'Age', data = df)\n",
    "plt.title('shcolarship V.S. age')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plotting having a scholarship against no show with respect to gender\n",
    "sns.barplot(x = 'Scholarship', y = 'No-show', hue = 'Gender', data = df)\n",
    "plt.title('shcolarship V.S. no show with respect to gender')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# ploting age destribution\n",
    "df['Age'].hist()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- we can see that having a scolarship does not affect showing up to a doctor appointment that much and that huge age group is enrolled to that scholarship and also enrol their babies on."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "___\n",
    "### **Does having certain deseas affects whather or not a patient may show up to their appointment? is it affected by gender?**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA4MAAAJKCAYAAACWB5AyAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAABjmElEQVR4nO3deXxUhdn3/+8kQwbMJIa4VFMWE5aniWwNuQloSF14RO2tthZMwEYpWCuVaChgFkhYRJYioQWqINTbCsqSEquP1V+rUZoSJOAS0ZiKIAZCwlIiNRPIZJnz+8MXczdFw0zIZDKcz/sv5sw1w3XmdZiL79nGYhiGIQAAAACAqQT5uwEAAAAAQOcjDAIAAACACREGAQAAAMCECIMAAAAAYEKEQQAAAAAwIau/G/ClsrIy2Ww2f7cBAPAxp9OpYcOG+buNgMF8BADzaGtGXtRh0GazKTY21t9tAAB8rKKiwt8tBBTmIwCYR1szktNEAQAAAMCECIMAAAAAYEKEQQAAAAAwoYv6mkEAMIOmpiZVVVWpoaHB3634XPfu3dWrVy9169bN360AALo4M81HqX0zkjAIAAGuqqpKYWFhuuaaa2SxWPzdjs8YhqGTJ0+qqqpK0dHR/m4HANDFmWU+Su2fkZwmCgABrqGhQZdddtlFP+gsFosuu+wy0+zhBQBcGLPMR6n9M5IwCAAXATMMOsk86wkA6BhmmhvtWVfCIAAAAACYkE/CYGFhodLS0pSWlqZ77rlHgwcPVllZmcaPH6/U1FStXr1akuRyuZSXl6eUlBSlpaWpsrJSkryqBQB4rrS0VMOHD1dNTY172ZNPPqnCwsJ2v+f111/fEa0BAOA3Zp2PPgmDd999tzZs2KANGzbo2muv1Zw5czR37lwtX75cmzZt0ocffqhPPvlEb775phobG7VlyxbNmDFDS5YskSSvagEA3gkJCVF2drYMw/B3KwAAdBlmnI8+vZvoRx99pP3792vGjBl67rnn1KdPH0lSUlKSdu7cqRMnTmj06NGSpGHDhunjjz+Ww+FQY2OjR7XouoxmpyxWm7/bCBh8XuhMI0eOlMvl0gsvvKCf/vSn7uXPPvus/vznP8tqtSohIUGzZs1q9Tqn06lHH31UDodDZ86c0fTp05WUlKTGxkbNmDFD1dXVioiI0MqVK3XmzBnNmjVLDodDLS0tevTRR1VfX6+dO3cqLy9PzzzzjN5//32tWbNGr7zyiqqrq/XQQw919kcBdDq+773HZ4bOYsb56NMwuHbtWj388MNyOByy2+3u5aGhoTp8+PA5y4ODg72qbW5ultX67avgdDpVUVHRwWsFT8TGxurQgsH+biNg9Mn7iG0V7dbU1KQzZ854VOt0OtXS0qLs7Gz99Kc/1X/913+publZp06d0p///Gf9z//8j6xWq2bMmKG//OUvSk5Odr92//79OnnypJ566inV1taqsrJSZ86c0enTpzV16lR997vf1ZQpU/TBBx/ojTfe0H/913/p3nvv1bFjx/Szn/1MhYWF+s1vfqMzZ85o165dqq2tVV1dnd544w1NnTrV43Voamri3wsClsVqYz56qU/eR/5uASYyb948jR8/3n0Qqr6+Xq+//ro2b94sq9Wq9PR0vf3227rxxhvdrzl06JBOnTql9evX6+TJk/riiy8kSadPn9b06dPVq1cvpaWlqaKiQq+//rquu+463X///Tp27JgmTJig1157Tb/97W8lSXv27NHJkyfV3Nyst956S+np6T5dX5+Fwa+++koHDx7UyJEj5XA4VF9f736uvr5e4eHhamhoaLXc5XLJbrd7XNtWEJQkm82m2NjYDlwrwHfYVtFeFRUV6tGjh0e1NptNwcHBuvrqqzV79mzNnTtX8fHxcrlc+v73v6/w8HBJUmJioiorK7Vw4UIdOnRIPXv21MqVKzVx4kTNnj1bzc3NSktLU48ePXTppZeqf//+kqTvfOc7MgxDlZWV+vGPf6wePXrommuuUVhYmFpaWhQTE6PPPvtMNptN3//+9/Xxxx/r+PHjiouL83h9u3Xrds6/l0AJh01NTcrKytKRI0cUFBSkxx9/XFarVVlZWbJYLBowYIDmzp2roKAgrV69Wtu3b5fValVOTo6GDBmiyspKj2sBAN7r2bOncnJylJmZqfj4eDmdTg0dOtT9Q+4JCQn67LPP9Oabb7aajykpKfrVr37lno+SdOmll6pXr16SpMsvv1xnzpzRgQMHdMcdd0j6emba7XY5HA5FR0dr7969slqtGjp0qPbs2aOamhr169fPp+vrs7uJ7tmzR6NGjZIk2e12devWTYcOHZJhGNqxY4cSEhIUHx+v4uJiSV/fNGbgwIFe1QIA2u+mm25SdHS0XnrpJdlsNu3du1fNzc0yDEN7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XStWrNDNN9+soUOH6tChQxowYICeeOIJ1dfXa/bs2fryyy8lSXPmzNH/+T//RzfeeKNiYmLUr18/FRcXq6GhQd///vfVq1cvLVy4UJIUERGhRYsW6ZNPPtG6devUrVs3VVVV6fbbb9fUqVP117/+VevWrZPVatWVV16pFStW6He/+50uv/xyTZgwQUuWLNF7770nSfrv//5v3X///crKylJISIiOHDmi48ePa8mSJbr22mv99tkBAMwnNDRUKSkpWrBgga666iqtWLFCGzdu1F//+ledOXNGPXv2dB8xLCsr0/333y+Hw6H09HTdcMMN2r17t1asWKHg4GD17t1bCxYs0Jo1a7R//36tXr1a999//zfO3uzsbFVWVqqhoUH33XeffvSjH/nxUwA6BmEQ6CS7du1SWlqa+/Hhw4f1yCOPuB8fO3ZMjz76qPr27atHH31Ub775pj788EONHDlSEydO1BdffKHs7Gxt2rRJNTU1KiwsVM+ePfW9731Pn3/+uW6++Wbdc889WrRokfr376+CggKtX79e1113naqrq/XKK6+osbFRo0eP1tSpU/Xqq69qypQpuvXWW/WnP/1JDofD3cvbb7+tqqoqbd26Vc3NzZo4caJGjhwpSYqKitKCBQu0detWbdmyRQsWLOi8DxEAAEmXXXaZvvzyS1111VVyuVw6deqUnnvuOQUFBWnKlCn66KOPJEk9evTQM888o9raWo0fP16jR49Wbm6uXnzxRV122WX6zW9+o5deekkPPfSQ9u3bp2nTpmnZsmXnzN5169Zpz5492rp1qySppKTEn6sPdBjCINBJRo4cqRUrVrgf/+d1DFdffbX69u0rSfr+97+vgwcPat++fdq1a5def/11SdK//vUvSVLPnj3Vs2fPc/6OAwcOaP78+ZKkpqYmXXPNNZKkgQMHymq1ymq1qnv37pKk7OxsrV27Vhs3blRMTIzGjBnT6n0SEhJksVjUrVs3DR06VAcOHJAkxcbGSpKuuuqqDjs9FQAAb1RXV+vOO+/UZ599pqCgIHXr1k2/+tWvdMkll+jo0aNqbm6WJA0fPlwWi0WXXXaZwsLC9OWXX+r48ePKyMiQJDU0NOi6665r9d7fNHvtdrtycnKUm5srh8OhO++8s1PXF/AVwiDQRRw7dkwnTpzQFVdcoffff1933XWXamtrdeedd+qOO+7Q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777b/Vxtba0yMzNVV1cnwzC0dOlSRUZGatasWXI4HGppadGjjz6qUaNGSZLmzZunqqoqSdLq1at1ySWXKDs7W1VVVWppadHPfvYz3X777e73X7VqlS6//HKNHTtWGRkZMgxDTqdT8+fPV1hYmKZPn66rr75aVVVV+uEPf6jPPvtMn3zyiW644Qb96le/6twPCgCA/9DS0qLZs2fr6NGjOn78uG666SZNnz5dWVlZCgkJ0ZEjR3T8+HEtWbJE1157rV544QUVFBToiiuu0MmTJyVJDQ0Nys7OVnV1tZqampSbm6sBAwZo9uzZqqur0/HjxzVx4kRNnDhRaWlpio6O1sGDB2UYhlasWKErrrjCz58CcOEIg4CfFBQUaPz48YqJiVFISIg+/PBD93NPPfWUbrrpJk2YMEHvv/++9u7dq4qKCl133XW6//77dezYMU2YMMF9qudPfvITJSQkKCsrSyUlJaqtrVVkZKSefPJJORwO3X333Ro5cuQ5Pezdu1cRERH69a9/rf379+v06dMKCwvT4cOH9eyzz6qhoUE333yziouL1aNHD914442EQQBAp9q1a5fS0tLcjw8fPqxHHnlEw4YN0/jx4+V0OpWcnOw+nTQqKkoLFizQ1q1btWXLFj3yyCN6/vnn9f/+3/+TxWJx73zdvHmzvvvd72rFihX64osvtH37doWEhOiHP/yhbrnlFh07dkxpaWmaOHGiJCk+Pl4LFizQCy+8oLVr12rOnDmd/2EAHYwwCPjBv/71LxUXF6u2tlYbNmyQw+HQxo0b3c8fPHhQ48aNk/T18ImPj9err76qO+64Q5L0ne98R3a73b13c9CgQZKkyy+/XA0NDTpw4ICuu+46SZLdble/fv10+PDhc/pITk7WF198oV/+8peyWq2aOnWqJKl3794KCwtTSEiILr/8ckVEREiSLBaLbz4QAAC+xciRI7VixQr347M7Ovfv369du3bJbrersbHR/XxsbKwk6aqrrtL777+vQ4cOqX///goJCZEkDRkyRJL0+eefKzk5WZJ0zTXXaNKkSTp27Jj+8Ic/6K9//avsdruam5tb9SF9PZffeust36400Em4ZhDwg1deeUU/+clP9Oyzz+r3v/+9tm7d6j6iJ0n9+vXTRx99JEnas2ePli1bpn79+undd9+VJB07dkxfffXVt4a0f691OBzat2+fevXqdU4fpaWluvLKK/Xss89q6tSpys/P/8b3AwCgqwkLC9Py5cs1efJkNTQ0yDAMSefOsGuuuUb79+9XQ0ODWlpaVFFRIan1rD18+LBmzJihZ599VsOGDdOTTz6pW2+91f2ekvTxxx9Lkt5//33179+/M1YR8DmODAJ+UFBQoF//+tfuxz169NAtt9yiP/7xj5Kkhx56SDk5OXrllVckSYsWLVJYWJhycnL0l7/8RQ0NDVqwYIGs1m/+J3zPPfcoNzdXEyZMkNPp1LRp03TZZZedU/e9731Pv/rVr7Rp0yY1Nzfr4Ycf9sHaAgDQsYKDg/X3v/9dZWVlCgkJUd++fXX8+PFvrI2MjNTPf/5zpaamKjIyUj169JAkpaamKicnRz/96U/V0tKinJwc1dfXa+HChXrttdcUFham4OBg91HHl156Sc8995x69OjRaoYDgcxi/PsuDwAAAACtpKWlad68eerXr5+/WwE6FKeJAgAAAIAJcWQQAAAAAEyII4MAAAAAYEKEQQAAAAAwIcIgAAAAAJgQYRAAAAAATIgwCAAAAAAm9P8DFqmxyI7PrcIAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 1080x720 with 4 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plotting deseases against no show\n",
    "plt.figure(figsize=(15,10))\n",
    "plt.subplot(2,2,1)\n",
    "sns.countplot(x = 'Hipertension', data = df, hue= 'No-show')\n",
    "plt.subplot(2,2,2)\n",
    "sns.countplot(x = 'Diabetes', data = df, hue= 'No-show')\n",
    "plt.subplot(2,2,3)\n",
    "sns.countplot(x = 'Alcoholism', data = df, hue= 'No-show')\n",
    "plt.subplot(2,2,4)\n",
    "sns.countplot(x = 'Handcap', data = df, hue= 'No-show')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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H31Uf/nLX7R+/XpZrVZde+21ysjI0O9+9ztdffXVmjx5spYvX66PP/5YkvRv//ZvmjZtmlJSUhQcHKzjx4+rpKREy5cv1y233OKz9w4AEHjatGmjiRMnaunSperYsaMyMjK0efNm/eUvf1FVVZXat2/vOlO3b98+TZs2TeXl5Zo9e7Zuv/12ffTRR8rIyFBQUJAiIyO1dOlSvfDCC/ryyy/129/+VtOmTTtv701NTdWRI0dUXV2tBx98UPfdd58P3wXAPQhywCXIzc1VQkKC6+djx47p8ccfd/1cXFysOXPmqHPnzpozZ4527typ/fv3Ky4uTlOmTNHhw4eVmpqqbdu26eTJk8rKylL79u11880366uvvtIdd9yhCRMm6Omnn1a3bt30xhtvaMOGDRo0aJBOnDih7du3q6amRkOHDtWjjz6qt99+WzNnztTo0aP15ptvqry83FXLBx98oKKiImVmZqqurk5TpkxRXFycJOn666/X0qVLlZmZqddff11Lly713psIAICkDh066JtvvlHHjh3lcDh05swZbdy4US1atNDMmTP12WefSZJatWqll156SadPn9b48eM1dOhQLV68WFu3blWHDh30n//5n/rjH/+oRx55RIWFhZo1a5ZWrlx5Tu9dv3699uzZo8zMTEnSrl27fLn5gNsQ5IBLEBcXp4yMDNfP/zpuv1OnTurcubMk6dZbb9WhQ4dUWFio3Nxcvfvuu5Kks2fPSpLat2+v9u3bn7OOgwcP6le/+pUkqba2Vl26dJEk3XjjjbJarbJarQoJCZEkpaam6sUXX9TmzZsVFRWlESNGNFhO//79ZbFY1LJlS/Xu3VsHDx6UJMXExEiSOnbs6LYhnQAANMWJEyd0zz336MCBA2rRooVatmypuXPnqnXr1vr6669VV1cnSerXr58sFos6dOigsLAwffPNNyopKVFiYqIkqbq6WoMGDWqw7PP13tDQUC1YsECLFy9WeXm57rnnHq9uL+ApBDnADYqLi3Xq1Cldc801+uSTT3Tvvffq9OnTuueeezR27FiVlpa6xu+3aPHD1NQWLVrI4XBIkrp27aoVK1bo+uuv18cff6xTp05JkiwWyznre/311zV79mx16NBBaWlpeu+991zPRUdHKysrS9OnT1dtba3+7//+T/fff/8FlwUAgLeUl5frjTfe0NSpUyVJ//jH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\n",
      "text/plain": [
       "<Figure size 1080x720 with 4 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# plotting deseases against no show with respect to age\n",
    "plt.figure(figsize=(15,10))\n",
    "plt.subplot(2,2,1)\n",
    "sns.barplot(x = 'Hipertension', y = 'Age', data = df, hue= 'No-show')\n",
    "plt.subplot(2,2,2)\n",
    "sns.barplot(x = 'Diabetes', y = 'Age', data = df, hue= 'No-show')\n",
    "plt.subplot(2,2,3)\n",
    "sns.barplot(x = 'Alcoholism', y = 'Age', data = df, hue= 'No-show')\n",
    "plt.subplot(2,2,4)\n",
    "sns.barplot(x = 'Handcap', y = 'Age', data = df, hue= 'No-show')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- from the previous set of plots, we can conclude that the vast majority of our dataset does not have chronic deseases, yet, they are existed in so many young people.  \n",
    "- having a chronic deseas may affect your showing up at a hospital's appointment."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "___\n",
    "## Conclusion\n",
    "\n",
    "### Q1: How often do men go to hospitals compared to women? Which of them is more likely to show up?\n",
    "- Nearly half of our dataset conists of women with wider age destribution and some outliers, all of which achiees a rate higher than men.\n",
    "\n",
    "- It is obvious that 79.8% of our patients did show up on their appointments and only 20.1% of them did not.\n",
    "\n",
    "- Women do show up on their appointments more often than men do, but this may b affected by the percentage of women on this dataset.\n",
    "___\n",
    "### Q2: Does recieving an SMS as a reminder affect whether or not a patient may show up? is it correlated with number of days before the appointment?\n",
    "- 67.8% of our patients did not reciee any SMS reminder of their appointments, yet they showed up on their appointments.\n",
    "- It is clear that there is a positive correlation between number of due days and whether a patient shows up or not.\n",
    "- Patient with appointments from 0 to 30 days tend to show up more regularly, while patients with higher number of days tend to not show up.\n",
    "- gender does not affect number of due days and showing up at an appointment that much.\n",
    "___\n",
    "### Q3: Does having a scholarship affects showing up on a hospital appointment? What are the age groups affected by this?\n",
    "- Having a scholarship does not affect showing up to a doctor appointment that much.\n",
    "- Huge age group is enrolled to that scholarship and also enrol their babies on.\n",
    "___\n",
    "### Q4: Does having certain deseases affect whather or not a patient may show up to their appointment? is it affected by gender?\n",
    "- We can conclude that the vast majority of our dataset does not have chronic deseases, yet, they are existed in so many young people.\n",
    "- Having a chronic deseas may affect your showing up at a hospital's appointment.\n"
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

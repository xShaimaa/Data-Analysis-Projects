# **Market Analysis Report for National Clothing Chain**

Project 3 of Udacity's [Data Analysis and Visualization with Microsoft Power BI Nanodegree Program](https://www.udacity.com/course/data-analysis-and-visualization-with-power-BI-nanodegree--nd331)
in **Advanced Data Analysis with Power BI** course.

## Project Description:
An online national clothing chain needs help on creating a targeted marketing campaign. 

Sales have been flat and they want to lure lost customers back. They want to advertise specific products to specific customers in specific locations, 
but they don’t know who to target. They have three products in mind:
- Shirt: $25
- Sweater: $100
- Leather Bag: $1,000
They need us to conduct an analysis to determine the best product to advertise to each customer.
___

## Data Sources
The project will use a variety of data sources, including
- US Census Bureau
  - Average income
  - location
  - population
  - industry

- Business Data
  - Product inventory
  - Product prices
  - Customer rating
  - Product return rate
  
- Customer Data
  - Customer ID
  - Names
  - Location
  - Date of birth
  - Purchase history
  
- Additional Data
  - Weather
  - Economics
  - Demographics
  - Competition
____

## Project Instruction
In this project, we will use population statistics from the US Census Bureau to determine where the greatest income exists around the country 
and whether there is a correlation between sales and income. We don’t know the incomes of our customers, but we should be able to predict it 
by looking at their purchase history and locations and comparing that against the census data. 
Additionally, we want to analyze our inventory, specifically customer ratings and return rate and see if there’s a correlation between the two.
___

## Data Model
A snapshot of the data model is provided below and can be found on `National-Clothing-Chain-Data-Model.png` on this repo.

![National Clothing Chain Data Model](https://github.com/xShaimaa/Udacity-Data-Analysis-and-Viz-with-Microsoft-Power-BI/blob/master/03-Market-Analysis-Report-for-National-Clothing-Chain/National-Clothing-Chain-Data-Model.png)


## Analysis Questions
1. What is the correlation (R2 value) between sales and income?
2. What is the correlation (R2 value) between customer ratings and product return rate?
3. What are the linear regression formulas to predict customer income from customer sales?
4. Which customer do you predict has the highest income?
5. Which product will be advertised the most?

Full report can be found on `National-Clothing-Chain-Report.pdf` and summery with finding can be found on `National-Clothing-Chain-Summary.doc` file, 
bot provided on this repo. The corresponding visuals can be seen grouped below.

![avg-income-by-state](https://github.com/xShaimaa/Udacity-Data-Analysis-and-Viz-with-Microsoft-Power-BI/blob/master/03-Market-Analysis-Report-for-National-Clothing-Chain/img/avg-income-by-state.png)
___
![predicted-income-by-state](https://github.com/xShaimaa/Udacity-Data-Analysis-and-Viz-with-Microsoft-Power-BI/blob/master/03-Market-Analysis-Report-for-National-Clothing-Chain/img/predicted-income-by-state.png)
___
![sales-income-corr](https://github.com/xShaimaa/Udacity-Data-Analysis-and-Viz-with-Microsoft-Power-BI/blob/master/03-Market-Analysis-Report-for-National-Clothing-Chain/img/sales-income-corr.png)
___
![customer-return-rate](https://github.com/xShaimaa/Udacity-Data-Analysis-and-Viz-with-Microsoft-Power-BI/blob/master/03-Market-Analysis-Report-for-National-Clothing-Chain/img/customer-return-rate.png)
___
![product-recomm](https://github.com/xShaimaa/Udacity-Data-Analysis-and-Viz-with-Microsoft-Power-BI/blob/master/03-Market-Analysis-Report-for-National-Clothing-Chain/img/product-recomm.png)
___
![product-by-price](https://github.com/xShaimaa/Udacity-Data-Analysis-and-Viz-with-Microsoft-Power-BI/blob/master/03-Market-Analysis-Report-for-National-Clothing-Chain/img/product-by-price.png)
___
![product-instock](https://github.com/xShaimaa/Udacity-Data-Analysis-and-Viz-with-Microsoft-Power-BI/blob/master/03-Market-Analysis-Report-for-National-Clothing-Chain/img/product-instock.png)
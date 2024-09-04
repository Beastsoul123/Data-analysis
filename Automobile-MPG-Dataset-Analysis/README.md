Auto-MPG Dataset Analysis:-

Contents

Dataset Description

Columns Descreption

Data Wrangling

Data Cleaning

Data Visualization

Conclusion

Built with



Dataset Description:

The MPG dataset is technical spec of cars originaly provided from UCI Machine Learning Repository and can be found on Kaggle here. The data concerns city-cycle fuel consumption in miles per gallon to be analyzed in terms of 3 multivalued discrete and 5 continuous attributes.

Columns Descreption:

mpg: miles per galon of fuel (continuous variable).

cylinders: number of engine cylinders (multi-valued discrete variable).

displacement: (continuous variable)

horsepower: the power produced by engine to move the car (continuous variable)

weight: car weight (continuous variable)

acceleration: the acceleration an engine can get per second (continuous variable)

model year: car release year from 1970 to 1982(multi-valued discrete variable)

origin: car manufacturing place (1 -> USA, 2 -> Europe, 3 -> Asia) (multi-valued discrete variable)

car name: car model name (unique for each instance)

Data Wrangling:

Our data can be found on auto-mpg.csv file provided on this repository, downloaded from Kaggle.

Data Cleaning:

Exploring Summary

Our dataset had a total of 398 records and 9 columns.

We had no NaNs in our dataset nor duplicated rows.

horsepower column had inconsistant data type that needed to be handled and casted to int.

origin needed to be parsed and casted into a categorical datatype.

No columns needed to be dropped.

Data Visualization

Using Matplotlib and Seaborn, we made several meaningful visuals and charts to help us gain informative insights regarding any correlation between attributes in our dataset, that'll be discussed in the next section.


Conclusion

These are derived conclusions after comleting our data visualisation phase.

As years pass after 1973, there has been a noticable increase in mpg.

As cylinders in the engine increases above 4, MPG decreases and engine horsepower increases. That indicates negative correlation between mpg and horsepower.

mpg increases as weight decreses over time, that also indecates a stron correlation between them.

Althogh USA has the biggest count of produced cars, its cars has relatively very low mpg, thus the highest possible weight compared to Asia and Europe

Asia is the leading contry in producing cars with high mpg with a mean close to 30, and it produces the lightest cars

Wa can spot a negative correlation between acceleration and horepower, this means that it has a positive one with mpg.

Built with:
Tools
JupyterLab
Python3
Pandas
Numpy
Matplotlib
Seaborn

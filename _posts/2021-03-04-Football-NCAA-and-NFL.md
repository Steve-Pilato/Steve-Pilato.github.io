---
title: "Who has Championship Games with Larger Point Differentials and More Blowouts: NCAA or NFL?"
date: 2021-03-04
tags: [webscraping, permutation testing, data science, R]
mathjax: "true"
---

### Hey everyone! So, I wanted to share an analysis I performed on NCAA and NFL championship data.

### To understand why I performed these analyses and made this notebook, it might help to have some context. So recently, my boss and I were discussing the 2020-2021 Super Bowl game. During our conversation, we had mentioned how surprised we were by the large point differential between the Tampa Bay Buccaneers and Kansas City Chiefs. Tampa Bay beat Kansas by 22 points, which to me seemed like a relatively large point differential for a professional football game. Now I will admit that I tend to watch more college than professional football, and thus have more experience with NCAA games. In the NCAA you can have large point differentials in games (i.e. Clemson beat Alabama by 28 points during the 2018-2019 year). This may be a result of particular conferences monopolizing better players/recruiting classes due to reputation.

### All of this made me think that maybe the NCAA, on average, has larger point differentials in championship games than in professional football. Also, maybe it was possible the the NCAA had more blowout championship games than the NFL. So I wanted to test these hypotheses by gathering Super Bowl and NCAA championship data.

## Load libraries

```R
library(rvest) # For web scraping
library(tidyverse) # For Data cleaning
library(infer) #For Permutation testing
```

# NFL Data

## Webscrape Super Bowl data

### NFL Data obtained from - https://www.pro-football-reference.com/super-bowl/

```R
superbowl_df <- html_session('https://www.pro-football-reference.com/super-bowl/') %>%
  read_html() %>%
  html_node(xpath = '//*[@id="super_bowls"]') %>%
  html_table()

#Change column names
colnames(superbowl_df)[c(4, 6)] <- c('winning_team', 'losing_team')
head(superbowl_df)
```
<table cellspacing="0" class="table table-condensed"><thead><tr><th align="left" style="text-align: left; max-width: 9px; min-width: 9px; border-bottom-color: rgba(255, 255, 255, 0.18);"><div class="pagedtable-header-name">&nbsp;</div><div class="pagedtable-header-type">&nbsp;</div></th><th align="left" style="text-align: left; max-width: 99px; min-width: 99px; border-bottom-color: rgba(255, 255, 255, 0.18);"><div class="pagedtable-header-name">Date</div><div class="pagedtable-header-type">&lt;chr&gt;</div></th><th align="left" style="text-align: left; max-width: 81px; min-width: 81px; border-bottom-color: rgba(255, 255, 255, 0.18);"><div class="pagedtable-header-name">SB</div><div class="pagedtable-header-type">&lt;chr&gt;</div></th><th align="left" style="text-align: left; max-width: 180px; min-width: 180px; border-bottom-color: rgba(255, 255, 255, 0.18);"><div class="pagedtable-header-name">Winner</div><div class="pagedtable-header-type">&lt;chr&gt;</div></th><th align="right" style="text-align: right; max-width: 108px; min-width: 108px; border-bottom-color: rgba(255, 255, 255, 0.18);"><div class="pagedtable-header-name">winning_team</div><div class="pagedtable-header-type">&lt;int&gt;</div></th><th align="left" style="text-align: left; max-width: 180px; min-width: 180px; border-bottom-color: rgba(255, 255, 255, 0.18);"><div class="pagedtable-header-name">Loser</div><div class="pagedtable-header-type">&lt;chr&gt;</div></th><th style="cursor: pointer; vertical-align: middle; min-width: 5px; width: 5px; border-bottom-color: rgba(255, 255, 255, 0.18);"><div style="border-top: 5px solid transparent;border-bottom: 5px solid transparent;border-left: 5px solid;"></div></th></tr></thead><tbody><tr class="odd"><td align="left" style="text-align: left; max-width: 9px; min-width: 9px; border-bottom-color: rgba(255, 255, 255, 0.18);">1</td><td align="left" style="text-align: left; max-width: 99px; min-width: 99px; border-bottom-color: rgba(255, 255, 255, 0.18);">Feb 7, 2021</td><td align="left" style="text-align: left; max-width: 81px; min-width: 81px; border-bottom-color: rgba(255, 255, 255, 0.18);">LV (55)</td><td align="left" style="text-align: left; max-width: 180px; min-width: 180px; border-bottom-color: rgba(255, 255, 255, 0.18);">Tampa Bay Buccaneers</td><td align="right" style="text-align: right; max-width: 108px; min-width: 108px; border-bottom-color: rgba(255, 255, 255, 0.18);">31</td><td align="left" style="text-align: left; max-width: 180px; min-width: 180px; border-bottom-color: rgba(255, 255, 255, 0.18);">Kansas City Chiefs</td><td style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td></tr><tr class="even" style="background-color: rgba(255, 255, 255, 0.02);"><td align="left" style="text-align: left; max-width: 9px; min-width: 9px; border-bottom-color: rgba(255, 255, 255, 0.18);">2</td><td align="left" style="text-align: left; max-width: 99px; min-width: 99px; border-bottom-color: rgba(255, 255, 255, 0.18);">Feb 2, 2020</td><td align="left" style="text-align: left; max-width: 81px; min-width: 81px; border-bottom-color: rgba(255, 255, 255, 0.18);">LIV (54)</td><td align="left" style="text-align: left; max-width: 180px; min-width: 180px; border-bottom-color: rgba(255, 255, 255, 0.18);">Kansas City Chiefs</td><td align="right" style="text-align: right; max-width: 108px; min-width: 108px; border-bottom-color: rgba(255, 255, 255, 0.18);">31</td><td align="left" style="text-align: left; max-width: 180px; min-width: 180px; border-bottom-color: rgba(255, 255, 255, 0.18);">San Francisco 49ers</td><td style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td></tr><tr class="odd"><td align="left" style="text-align: left; max-width: 9px; min-width: 9px; border-bottom-color: rgba(255, 255, 255, 0.18);">3</td><td align="left" style="text-align: left; max-width: 99px; min-width: 99px; border-bottom-color: rgba(255, 255, 255, 0.18);">Feb 3, 2019</td><td align="left" style="text-align: left; max-width: 81px; min-width: 81px; border-bottom-color: rgba(255, 255, 255, 0.18);">LIII (53)</td><td align="left" style="text-align: left; max-width: 180px; min-width: 180px; border-bottom-color: rgba(255, 255, 255, 0.18);">New England Patriots</td><td align="right" style="text-align: right; max-width: 108px; min-width: 108px; border-bottom-color: rgba(255, 255, 255, 0.18);">13</td><td align="left" style="text-align: left; max-width: 180px; min-width: 180px; border-bottom-color: rgba(255, 255, 255, 0.18);">Los Angeles Rams</td><td style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td></tr><tr class="even" style="background-color: rgba(255, 255, 255, 0.02);"><td align="left" style="text-align: left; max-width: 9px; min-width: 9px; border-bottom-color: rgba(255, 255, 255, 0.18);">4</td><td align="left" style="text-align: left; max-width: 99px; min-width: 99px; border-bottom-color: rgba(255, 255, 255, 0.18);">Feb 4, 2018</td><td align="left" style="text-align: left; max-width: 81px; min-width: 81px; border-bottom-color: rgba(255, 255, 255, 0.18);">LII (52)</td><td align="left" style="text-align: left; max-width: 180px; min-width: 180px; border-bottom-color: rgba(255, 255, 255, 0.18);">Philadelphia Eagles</td><td align="right" style="text-align: right; max-width: 108px; min-width: 108px; border-bottom-color: rgba(255, 255, 255, 0.18);">41</td><td align="left" style="text-align: left; max-width: 180px; min-width: 180px; border-bottom-color: rgba(255, 255, 255, 0.18);">New England Patriots</td><td style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td></tr><tr class="odd"><td align="left" style="text-align: left; max-width: 9px; min-width: 9px; border-bottom-color: rgba(255, 255, 255, 0.18);">5</td><td align="left" style="text-align: left; max-width: 99px; min-width: 99px; border-bottom-color: rgba(255, 255, 255, 0.18);">Feb 5, 2017</td><td align="left" style="text-align: left; max-width: 81px; min-width: 81px; border-bottom-color: rgba(255, 255, 255, 0.18);">LI (51)</td><td align="left" style="text-align: left; max-width: 180px; min-width: 180px; border-bottom-color: rgba(255, 255, 255, 0.18);">New England Patriots</td><td align="right" style="text-align: right; max-width: 108px; min-width: 108px; border-bottom-color: rgba(255, 255, 255, 0.18);">34</td><td align="left" style="text-align: left; max-width: 180px; min-width: 180px; border-bottom-color: rgba(255, 255, 255, 0.18);">Atlanta Falcons</td><td style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td></tr><tr class="even" style="background-color: rgba(255, 255, 255, 0.02);"><td align="left" style="text-align: left; max-width: 9px; min-width: 9px; border-bottom-color: rgba(255, 255, 255, 0.18);">6</td><td align="left" style="text-align: left; max-width: 99px; min-width: 99px; border-bottom-color: rgba(255, 255, 255, 0.18);">Feb 7, 2016</td><td align="left" style="text-align: left; max-width: 81px; min-width: 81px; border-bottom-color: rgba(255, 255, 255, 0.18);">50</td><td align="left" style="text-align: left; max-width: 180px; min-width: 180px; border-bottom-color: rgba(255, 255, 255, 0.18);">Denver Broncos</td><td align="right" style="text-align: right; max-width: 108px; min-width: 108px; border-bottom-color: rgba(255, 255, 255, 0.18);">24</td><td align="left" style="text-align: left; max-width: 180px; min-width: 180px; border-bottom-color: rgba(255, 255, 255, 0.18);">Carolina Panthers</td><td style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td></tr></tbody></table>

## Filter data and calculate point differentials for Super Bowl

### Please note that I will only be looking at data from 1990 to 2019

```R
superbowl_filtered <- superbowl_df %>%
  mutate(year = as.integer(str_extract_all(Date, "[0-9]{4}")), point_differential =  winning_team - losing_team) %>%
  filter(year >= 1990 & year <= 2019) %>%
  select(year, point_differential)

```

## Calculate descriptive statistics

```R
summary(superbowl_filtered$point_differential)
```
<div class="GCHYANPCH5C" style="opacity: 1;"><div><pre data-ordinal="1" style="margin-top: 0px; white-space: pre-wrap;"><span>   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
   1.00    4.00   10.00   12.80   14.75   45.00 
</span></pre></div></div>

# NCAA Data

## Webscrape NCAA football National championship data

### NCAA Data obtained from - http://championshiphistory.com/ncaafootball.php
```R
NCAA_df <- html_session('http://championshiphistory.com/ncaafootball.php') %>%
  read_html() %>%
  html_node(xpath = '//*[@id="tablesorter-demo"]') %>%
  html_table(fill = T)

NCAA_df <- na.omit(NCAA_df)
head(NCAA_df)
```
<table cellspacing="0" class="table table-condensed"><thead><tr><th align="left" style="text-align: left; max-width: 18px; min-width: 18px; border-bottom-color: rgba(255, 255, 255, 0.18);"><div class="pagedtable-header-name">&nbsp;</div><div class="pagedtable-header-type">&nbsp;</div></th><th align="right" style="text-align: right; max-width: 36px; min-width: 36px; border-bottom-color: rgba(255, 255, 255, 0.18);"><div class="pagedtable-header-name">Year</div><div class="pagedtable-header-type">&lt;int&gt;</div></th><th align="left" style="text-align: left; max-width: 27px; min-width: 27px; border-bottom-color: rgba(255, 255, 255, 0.18);"><div class="pagedtable-header-name">Era</div><div class="pagedtable-header-type">&lt;chr&gt;</div></th><th align="left" style="text-align: left; max-width: 126px; min-width: 126px; border-bottom-color: rgba(255, 255, 255, 0.18);"><div class="pagedtable-header-name">#1 CollegeTeam</div><div class="pagedtable-header-type">&lt;chr&gt;</div></th><th align="left" style="text-align: left; max-width: 72px; min-width: 72px; border-bottom-color: rgba(255, 255, 255, 0.18);"><div class="pagedtable-header-name">RunnerUp</div><div class="pagedtable-header-type">&lt;chr&gt;</div></th><th align="left" style="text-align: left; max-width: 162px; min-width: 162px; border-bottom-color: rgba(255, 255, 255, 0.18);"><div class="pagedtable-header-name">Championship Score</div><div class="pagedtable-header-type">&lt;chr&gt;</div></th><th style="cursor: pointer; vertical-align: middle; min-width: 5px; width: 5px; border-bottom-color: rgba(255, 255, 255, 0.18);"><div style="border-top: 5px solid transparent;border-bottom: 5px solid transparent;border-left: 5px solid;"></div></th></tr></thead><tbody><tr class="odd"><td align="left" style="text-align: left; max-width: 18px; min-width: 18px; border-bottom-color: rgba(255, 255, 255, 0.18);">1</td><td align="right" style="text-align: right; max-width: 36px; min-width: 36px; border-bottom-color: rgba(255, 255, 255, 0.18);">2019</td><td align="left" style="text-align: left; max-width: 27px; min-width: 27px; border-bottom-color: rgba(255, 255, 255, 0.18);">CFP</td><td align="left" style="text-align: left; max-width: 126px; min-width: 126px; border-bottom-color: rgba(255, 255, 255, 0.18);">LSU Tigers</td><td align="left" style="text-align: left; max-width: 72px; min-width: 72px; border-bottom-color: rgba(255, 255, 255, 0.18);">Clemson</td><td align="left" style="text-align: left; max-width: 162px; min-width: 162px; border-bottom-color: rgba(255, 255, 255, 0.18);">42-25</td><td style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td></tr><tr class="even" style="background-color: rgba(255, 255, 255, 0.02);"><td align="left" style="text-align: left; max-width: 18px; min-width: 18px; border-bottom-color: rgba(255, 255, 255, 0.18);">3</td><td align="right" style="text-align: right; max-width: 36px; min-width: 36px; border-bottom-color: rgba(255, 255, 255, 0.18);">2018</td><td align="left" style="text-align: left; max-width: 27px; min-width: 27px; border-bottom-color: rgba(255, 255, 255, 0.18);">CFP</td><td align="left" style="text-align: left; max-width: 126px; min-width: 126px; border-bottom-color: rgba(255, 255, 255, 0.18);">Clemson</td><td align="left" style="text-align: left; max-width: 72px; min-width: 72px; border-bottom-color: rgba(255, 255, 255, 0.18);">Alabama</td><td align="left" style="text-align: left; max-width: 162px; min-width: 162px; border-bottom-color: rgba(255, 255, 255, 0.18);">44-16</td><td style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td></tr><tr class="odd"><td align="left" style="text-align: left; max-width: 18px; min-width: 18px; border-bottom-color: rgba(255, 255, 255, 0.18);">5</td><td align="right" style="text-align: right; max-width: 36px; min-width: 36px; border-bottom-color: rgba(255, 255, 255, 0.18);">2017</td><td align="left" style="text-align: left; max-width: 27px; min-width: 27px; border-bottom-color: rgba(255, 255, 255, 0.18);">CFP</td><td align="left" style="text-align: left; max-width: 126px; min-width: 126px; border-bottom-color: rgba(255, 255, 255, 0.18);">Alabama</td><td align="left" style="text-align: left; max-width: 72px; min-width: 72px; border-bottom-color: rgba(255, 255, 255, 0.18);">Georgia</td><td align="left" style="text-align: left; max-width: 162px; min-width: 162px; border-bottom-color: rgba(255, 255, 255, 0.18);">26-23</td><td style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td></tr><tr class="even" style="background-color: rgba(255, 255, 255, 0.02);"><td align="left" style="text-align: left; max-width: 18px; min-width: 18px; border-bottom-color: rgba(255, 255, 255, 0.18);">7</td><td align="right" style="text-align: right; max-width: 36px; min-width: 36px; border-bottom-color: rgba(255, 255, 255, 0.18);">2016</td><td align="left" style="text-align: left; max-width: 27px; min-width: 27px; border-bottom-color: rgba(255, 255, 255, 0.18);">CFP</td><td align="left" style="text-align: left; max-width: 126px; min-width: 126px; border-bottom-color: rgba(255, 255, 255, 0.18);">Clemson</td><td align="left" style="text-align: left; max-width: 72px; min-width: 72px; border-bottom-color: rgba(255, 255, 255, 0.18);">Alabama</td><td align="left" style="text-align: left; max-width: 162px; min-width: 162px; border-bottom-color: rgba(255, 255, 255, 0.18);">35-31</td><td style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td></tr><tr class="odd"><td align="left" style="text-align: left; max-width: 18px; min-width: 18px; border-bottom-color: rgba(255, 255, 255, 0.18);">9</td><td align="right" style="text-align: right; max-width: 36px; min-width: 36px; border-bottom-color: rgba(255, 255, 255, 0.18);">2015</td><td align="left" style="text-align: left; max-width: 27px; min-width: 27px; border-bottom-color: rgba(255, 255, 255, 0.18);">CFP</td><td align="left" style="text-align: left; max-width: 126px; min-width: 126px; border-bottom-color: rgba(255, 255, 255, 0.18);">Alabama</td><td align="left" style="text-align: left; max-width: 72px; min-width: 72px; border-bottom-color: rgba(255, 255, 255, 0.18);">Clemson</td><td align="left" style="text-align: left; max-width: 162px; min-width: 162px; border-bottom-color: rgba(255, 255, 255, 0.18);">45-40</td><td style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td></tr><tr class="even" style="background-color: rgba(255, 255, 255, 0.02);"><td align="left" style="text-align: left; max-width: 18px; min-width: 18px; border-bottom-color: rgba(255, 255, 255, 0.18);">11</td><td align="right" style="text-align: right; max-width: 36px; min-width: 36px; border-bottom-color: rgba(255, 255, 255, 0.18);">2014</td><td align="left" style="text-align: left; max-width: 27px; min-width: 27px; border-bottom-color: rgba(255, 255, 255, 0.18);">CFP</td><td align="left" style="text-align: left; max-width: 126px; min-width: 126px; border-bottom-color: rgba(255, 255, 255, 0.18);">Ohio State</td><td align="left" style="text-align: left; max-width: 72px; min-width: 72px; border-bottom-color: rgba(255, 255, 255, 0.18);">Oregon</td><td align="left" style="text-align: left; max-width: 162px; min-width: 162px; border-bottom-color: rgba(255, 255, 255, 0.18);">42-20</td><td style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td></tr></tbody></table>

## Parse out championship scores for NCAA

```R
# Calculate number of rows in NCAA data frame
num_games <- nrow(NCAA_df)

#Create variable for storing winning team point results
winning_team <- NULL

#Create variable for storing losing team point results
losing_team <- NULL

# Create variable for storing year
year <- NULL
for(curr_game in 1:num_games){
  
  #Create if statement for detecting rows in championship score column without scores
  if(str_length(NCAA_df$`Championship Score`[curr_game]) != 0){
  scores <- str_split(NCAA_df$`Championship Score`[curr_game], "-")[[1]]
  winning_team[curr_game] <- scores[1]
  losing_team[curr_game] <- scores[2]
  year[curr_game] <- NCAA_df$Year[curr_game]
  }
  #If there are no scores present skip iteration
  else{
    next
  }

}

# Create data frame from variables that had data appended to them in for loop
NCAA_scores <- data.frame(year = unlist(year), winning_score = as.integer(unlist(winning_team)), losing_score = as.integer(unlist(losing_team)))

```

## Calculate NCAA point differentials

```R
NCAA_filtered <- NCAA_scores %>%
  filter(year >= 1990 & year <= 2019) %>%
  mutate(point_differential =  winning_score - losing_score) %>%
  select(year, point_differential)

head(NCAA_filtered)
```

## Calculate descriptive statistics

```R
summary(NCAA_filtered$point_differential)
```

<pre data-ordinal="1" style="margin-top: 0px; white-space: pre-wrap;"><span>   Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
   1.00    5.00   12.50   14.67   22.00   38.00 
</span></pre>

# Combine Superbowl and NCAA filtered data frames for data visualization

```R
# Add Super Bowl Label
superbowl_filtered$organization <- "NFL"

# Add NCAA Label
NCAA_filtered$organization <- "NCAA"

#Combine data
all_data <- rbind(superbowl_filtered, NCAA_filtered)
head(all_data)
```

## Graph data

```R
# Create boxplot
ggplot(data = all_data, aes(x = organization, y = point_differential, color = organization)) +
  geom_boxplot(lwd = 1) +
  geom_jitter() +
  ggtitle("Distribution of Super Bowl and NCAA Championship Point Differentials") +
  ylab("Point Differential") +
  xlab("Organization") +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(), 
        axis.line = element_line(colour = 'black'), 
        title = element_text(hjust = 0.5, face = 'bold'),
        axis.text.x = element_text(face = 'bold', size = 10, color = 'black'),
        axis.text.y = element_text(face = 'bold', size = 10, color = 'black'),
        axis.title = element_text(color = 'black', face = 'bold'))

```

![png](/images/blowout_images/blowouts_box.png)



### From the above figure, we can see that NFL Super Bowl games have a slightly lower median compared to the NCAA games. It is important to realize that this does not mean that there is statistically significant difference between the two groups. In order to determine if this difference is meaningful we will need to perform some inferential statistics.

### We can use permutation testing to see if there is a meaningful difference between the means of the NFL and NCAA point differentials. Briefly, permutation testing allows us to create a null distribution of a statistic of interest (In this case the difference in average point differentials) by shuffling the labels (NCAA and NFL) belonging to the point differentials. We then can generate the mean difference from the resulting shuffle and do this repeatedly.

# Permutation Testing: Is there a meaningful difference in the average point differential between the NCAA and NFL championship games?

## Calculate observed mean difference in point differential between the NFL and NCAA games

```R
obs_diff <- all_data %>%
  specify(point_differential ~ organization) %>%
  calculate(stat = "diff in means", order = c("NCAA", "NFL"))

#Show observed difference  
obs_diff
```
<table cellspacing="0" class="table table-condensed"><thead><tr><th align="right" style="text-align: right; max-width: 72px; min-width: 72px; border-bottom-color: rgba(255, 255, 255, 0.18);"><div class="pagedtable-header-name">stat</div><div class="pagedtable-header-type">&lt;dbl&gt;</div></th><th class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></th><th class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></th><th class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></th><th class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></th></tr></thead><tbody><tr class="odd"><td align="right" style="text-align: right; max-width: 72px; min-width: 72px; border-bottom-color: rgba(255, 255, 255, 0.18);">1.866667</td><td class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td><td class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td><td class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td><td class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td></tr></tbody></table>
## Create Null distribution

```R
null_distrubution <- all_data %>%
  specify(point_differential ~ organization) %>%
  hypothesize(null = "independence") %>%
  generate(reps = 1000, type = "permute") %>%
  calculate(stat = "diff in means", order = c("NCAA", "NFL"))
```

## Visualize the null distribution and calculate p-value

```R
#visualize null distribution
visualise(null_distrubution, bins = 15) +
  shade_p_value(obs_stat = obs_diff, direction = "right")

```
![png](/images/blowout_images/blowouts_point_differential_null.png)

## Calculate p-value

```R
#Calculate p-value
null_distrubution %>%
  get_p_value(obs_stat = obs_diff, direction = 'right')
```
<table cellspacing="0" class="table table-condensed"><thead><tr><th align="right" style="text-align: right; max-width: 63px; min-width: 63px; border-bottom-color: rgba(255, 255, 255, 0.18);"><div class="pagedtable-header-name">p_value</div><div class="pagedtable-header-type">&lt;dbl&gt;</div></th><th class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></th><th class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></th><th class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></th><th class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></th></tr></thead><tbody><tr class="odd"><td align="right" style="text-align: right; max-width: 63px; min-width: 63px; border-bottom-color: rgba(255, 255, 255, 0.18);">0.246</td><td class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td><td class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td><td class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td><td class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td></tr></tbody></table>

### Since the p-value is greater 0.5, we fail to reject the null hypothesis that the average point differential for Super Bowl games is the same as the average point differential for NCAA championship games. Thus, there is no meaningful difference between the two means.

# Is There a Difference In the Proportion of "Blowout" Championship Games Between NCAA and NFL

### Though there does not seem to be any agreement on what is a blowout game, it looks like many individuals agree with a number around 17 and 20 plus points.

### I am going to make assumption that a 20 point differential is what constitutes a blowout game.

## Calculate the number of blowout games

```R
blowouts <- all_data %>%
  mutate(game_status = if_else(point_differential >= 20, true = "blowout", false = "non-blowout")) %>%
  group_by(organization) %>%
  count(game_status) 
  

```

## Visualize data

```R
ggplot(data = blowouts, aes(x = organization, y = n, fill = game_status)) +
  geom_col(color = 'black') +
    ggtitle("Number of Games that were blowout and non-blowout games") +
  ylab("Frequency") +
  xlab("Organization") +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(), 
        axis.line = element_line(colour = 'black'), 
        title = element_text(hjust = 0.5, face = 'bold'),
        axis.text.x = element_text(face = 'bold', size = 10, color = 'black'),
        axis.text.y = element_text(face = 'bold', size = 10, color = 'black'),
        axis.title = element_text(color = 'black', face = 'bold'))
```

![png](/images/blowout_images/proportions.png)

### Based on the above figure, there seems to be a larger proportion of NCAA championship games that ended up with a blowout game.

### I thought it would be a good idea to calculate the actual percentage of games that were blowouts for both organizations.

## Calculate difference in proportion of blowout game between NFL and NCAA Games

```R
blowouts %>%
  group_by(organization) %>%
  mutate(group_total = sum(n)) %>%
  ungroup() %>%
  group_by(organization, game_status) %>%
  summarise(percentage = round((n/group_total)*100, 2)) %>%
  filter(game_status == "blowout")

```
<table cellspacing="0" class="table table-condensed"><thead><tr><th align="left" style="text-align: left; max-width: 108px; min-width: 108px; border-bottom-color: rgba(255, 255, 255, 0.18);"><div class="pagedtable-header-name">organization</div><div class="pagedtable-header-type">&lt;chr&gt;</div></th><th align="left" style="text-align: left; max-width: 99px; min-width: 99px; border-bottom-color: rgba(255, 255, 255, 0.18);"><div class="pagedtable-header-name">game_status</div><div class="pagedtable-header-type">&lt;chr&gt;</div></th><th align="right" style="text-align: right; max-width: 90px; min-width: 90px; border-bottom-color: rgba(255, 255, 255, 0.18);"><div class="pagedtable-header-name">percentage</div><div class="pagedtable-header-type">&lt;dbl&gt;</div></th><th class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></th><th class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></th></tr></thead><tbody><tr class="odd"><td align="left" style="text-align: left; max-width: 108px; min-width: 108px; border-bottom-color: rgba(255, 255, 255, 0.18);">NCAA</td><td align="left" style="text-align: left; max-width: 99px; min-width: 99px; border-bottom-color: rgba(255, 255, 255, 0.18);">blowout</td><td align="right" style="text-align: right; max-width: 90px; min-width: 90px; border-bottom-color: rgba(255, 255, 255, 0.18);">36.67</td><td class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td><td class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td></tr><tr class="even" style="background-color: rgba(255, 255, 255, 0.02);"><td align="left" style="text-align: left; max-width: 108px; min-width: 108px; border-bottom-color: rgba(255, 255, 255, 0.18);">NFL</td><td align="left" style="text-align: left; max-width: 99px; min-width: 99px; border-bottom-color: rgba(255, 255, 255, 0.18);">blowout</td><td align="right" style="text-align: right; max-width: 90px; min-width: 90px; border-bottom-color: rgba(255, 255, 255, 0.18);">20.00</td><td class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td><td class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td></tr></tbody></table>

### So, we can see that the proportion of games that I considered "blowouts" for the NCAA and NFL championship games was 36.67% and 20% respectively

### This means that blowout games occurred 16.67% more in NCAA title games than in Super Bowl games in our sample.

### Again, the question is whether or not this is statistically meaningful.

# Permutation Testing: Is there a meaningful difference in the Proportion of Championship blowout games Between NCAA and NFL?

## Calculate the observed difference in proportion of championship blowout games between NCAA and NFL

```R
#create new df
blowouts <- all_data %>%
  mutate(game_status = if_else(point_differential >= 20, true = "blowout", false = "non-blowout"))

#Calculate obs difference
obs_diff <- blowouts %>%
  specify(game_status ~ organization, success = "blowout") %>%
  calculate(stat = "diff in props", order = c("NCAA", "NFL"))

#Show observed difference  
obs_diff
```
<table cellspacing="0" class="table table-condensed"><thead><tr><th align="right" style="text-align: right; max-width: 81px; min-width: 81px; border-bottom-color: rgba(255, 255, 255, 0.18);"><div class="pagedtable-header-name">stat</div><div class="pagedtable-header-type">&lt;dbl&gt;</div></th><th class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></th><th class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></th><th class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></th><th class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></th></tr></thead><tbody><tr class="odd"><td align="right" style="text-align: right; max-width: 81px; min-width: 81px; border-bottom-color: rgba(255, 255, 255, 0.18);">0.1666667</td><td class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td><td class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td><td class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td><td class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td></tr></tbody></table>

## Create Null distribution

```{r}
null_distrubution <- blowouts %>%
  specify(game_status ~ organization, success = "blowout") %>%
  hypothesize(null = "independence") %>%
  generate(reps = 1000, type = "permute") %>%
  calculate(stat = "diff in props", order = c("NCAA", "NFL"))
```

## Visualize the null distribution and calculate p-value

```R
set.seed(1234)

#visualize null distribution
visualise(null_distrubution, bins = 10) +
  shade_p_value(obs_stat = obs_diff, direction = "right")

```
![png](/images/blowout_images/blowouts_differentce_blowouts_null.png)

## Calculate p-value

```R
#Calculate p-value
null_distrubution %>%
  get_p_value(obs_stat = obs_diff, direction = 'right')
```
<table cellspacing="0" class="table table-condensed"><thead><tr><th align="right" style="text-align: right; max-width: 63px; min-width: 63px; border-bottom-color: rgba(255, 255, 255, 0.18);"><div class="pagedtable-header-name">p_value</div><div class="pagedtable-header-type">&lt;dbl&gt;</div></th><th class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></th><th class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></th><th class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></th><th class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></th></tr></thead><tbody><tr class="odd"><td align="right" style="text-align: right; max-width: 63px; min-width: 63px; border-bottom-color: rgba(255, 255, 255, 0.18);">0.121</td><td class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td><td class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td><td class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td><td class="pagedtable-padding-col" style="border-bottom-color: rgba(255, 255, 255, 0.18);"></td></tr></tbody></table>

### As with the point differential section, we fail to reject the null hypothesis *p* > 0.5.

# Concluding Remarks

### Well, this was a really fun experiment and definitely shows why it is important to perform these kinds of analyses. We can all be subject to the availability heuristic, which makes us evaluate certain topics based on examples that come readily to our minds. I felt really confident that college championship games would more likely have larger point differentials or even more blowout games, but this was not the case.

### Hope you enjoyed this notebook and feel free to email me if you have any questions about these data or the code!

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

```{r}
.libPaths('D:/R_3.6.1_libraries') #change library path
library(rvest) # For web scraping
library(tidyverse) # For Data cleaning
library(infer) #For Permutation testing
```

# NFL Data

## Webscrape Super Bowl data

```{r}
superbowl_df <- html_session('https://www.pro-football-reference.com/super-bowl/') %>%
  read_html() %>%
  html_node(xpath = '//*[@id="super_bowls"]') %>%
  html_table()

#Change column names
colnames(superbowl_df)[c(4, 6)] <- c('winning_team', 'losing_team')
head(superbowl_df)
```

## Filter data and calculate point differentials for Super Bowl

### Please note that I will only be looking at data from 1990 to 2019

```{r}
superbowl_filtered <- superbowl_df %>%
  mutate(year = as.integer(str_extract_all(Date, "[0-9]{4}")), point_differential =  winning_team - losing_team) %>%
  filter(year >= 1990 & year <= 2019) %>%
  select(year, point_differential)

```

## Calculate descriptive statistics

```{r}
summary(superbowl_filtered$point_differential)
```

# NCAA Data

## Webscrape NCAA football National championship data

```{r}
NCAA_df <- html_session('http://championshiphistory.com/ncaafootball.php') %>%
  read_html() %>%
  html_node(xpath = '//*[@id="tablesorter-demo"]') %>%
  html_table(fill = T)

NCAA_df <- na.omit(NCAA_df)
head(NCAA_df)
```

## Parse out championship scores for NCAA

```{r}
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

```{r}
NCAA_filtered <- NCAA_scores %>%
  filter(year >= 1990 & year <= 2019) %>%
  mutate(point_differential =  winning_score - losing_score) %>%
  select(year, point_differential)

head(NCAA_filtered)
```

## Calculate descriptive statistics

```{r}
summary(NCAA_filtered$point_differential)
```

# Combine Superbowl and NCAA filtered data frames for data visualization

```{r}
# Add Super Bowl Label
superbowl_filtered$organization <- "NFL"

# Add NCAA Label
NCAA_filtered$organization <- "NCAA"

#Combine data
all_data <- rbind(superbowl_filtered, NCAA_filtered)
head(all_data)
```

## Graph data

```{r}
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

### From the above figure, we can see that NFL Super Bowl games have a slightly lower median compared to the NCAA games. It is important to realize that this does not mean that there is statistically significant difference between the two groups. In order to determine if this difference is meaningful we will need to perform some inferential statistics.

### We can use permutation testing to see if there is a meaningful difference between the means of the NFL and NCAA point differentials. Briefly, permutation testing allows us to create a null distribution of a statistic of interest (In this case the difference in average point differentials) by shuffling the labels (NCAA and NFL) belonging to the point differentials. We then can generate the mean difference from the resulting shuffle and do this repeatedly.

# Permutation Testing: Is there a meaningful difference in the average point differential between the NCAA and NFL championship games?

## Calculate observed mean difference in point differential between the NFL and NCAA games

```{r}
obs_diff <- all_data %>%
  specify(point_differential ~ organization) %>%
  calculate(stat = "diff in means", order = c("NCAA", "NFL"))

#Show observed difference  
obs_diff
```

## Create Null distribution

```{r}
null_distrubution <- all_data %>%
  specify(point_differential ~ organization) %>%
  hypothesize(null = "independence") %>%
  generate(reps = 1000, type = "permute") %>%
  calculate(stat = "diff in means", order = c("NCAA", "NFL"))
```

## Visualize the null distribution and calculate p-value

```{r}
#visualize null distribution
visualise(null_distrubution, bins = 15) +
  shade_p_value(obs_stat = obs_diff, direction = "right")

```

## Calculate p-value

```{r}
#Calculate p-value
null_distrubution %>%
  get_p_value(obs_stat = obs_diff, direction = 'right')
```

### Since the p-value is greater 0.5, we fail to reject the null hypothesis that the average point differential for Super Bowl games is the same as the average point differential for NCAA championship games. Thus, there is no meaningful difference between the two means.

# Is There a Difference In the Proportion of "Blowout" Championship Games Between NCAA and NFL

### Though there does not seem to be any agreement on what is a blowout game, it looks like many individuals agree with a number around 17 and 20 plus points.

### I am going to make assumption that a 20 point differential is what constitutes a blowout game.

## Calculate the number of blowout games

```{r}
blowouts <- all_data %>%
  mutate(game_status = if_else(point_differential >= 20, true = "blowout", false = "non-blowout")) %>%
  group_by(organization) %>%
  count(game_status) 
  

```

## Visualize data

```{r}
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

### Based on the above figure, there seems to be a larger proportion of NCAA championship games that ended up with a blowout game.

### I thought it would be a good idea to calculate the actual percentage of games that were blowouts for both organizations.

## Calculate difference in proportion of blowout game between NFL and NCAA Games

```{r}
blowouts %>%
  group_by(organization) %>%
  mutate(group_total = sum(n)) %>%
  ungroup() %>%
  group_by(organization, game_status) %>%
  summarise(percentage = round((n/group_total)*100, 2)) %>%
  filter(game_status == "blowout")

```

### So, we can see that the proportion of games that I considered "blowouts" for the NCAA and NFL championship games was 36.67% and 20% respectively

### This means that blowout games occurred 16.67% more in NCAA title games than in Super Bowl games in our sample.

### Again, the question is whether or not this is statistically meaningful.

# Permutation Testing: Is there a meaningful difference in the Proportion of Championship blowout games Between NCAA and NFL?

## Calculate the observed difference in proportion of championship blowout games between NCAA and NFL

```{r}
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

## Create Null distribution

```{r}
null_distrubution <- blowouts %>%
  specify(game_status ~ organization, success = "blowout") %>%
  hypothesize(null = "independence") %>%
  generate(reps = 1000, type = "permute") %>%
  calculate(stat = "diff in props", order = c("NCAA", "NFL"))
```

## Visualize the null distribution and calculate p-value

```{r}
set.seed(1234)

#visualize null distribution
visualise(null_distrubution, bins = 10) +
  shade_p_value(obs_stat = obs_diff, direction = "right")

```

## Calculate p-value

```{r}
#Calculate p-value
null_distrubution %>%
  get_p_value(obs_stat = obs_diff, direction = 'right')
```

### As with the point differential section, we fail to reject the null hypothesis *p* \> 0.5.

# Concluding Remarks

### Well, this was a really fun experiment and definitely shows why it is important to perform these kinds of analyses. We can all be subject to the availability heuristic, which makes us evaluate certain topics based on examples that come readily to our minds. I felt really confident that college championship games would more likely have larger point differentials or even more blowout games, but this was not the case.

### Hope you enjoyed this notebook and feel free to email me if you have any questions about these data or the code!

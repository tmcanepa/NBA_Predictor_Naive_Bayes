import pandas as pd
import numpy as np
import sys



if len(sys.argv) != 3:
    print("Needs 3 arguments")
    sys.exit(1)

train_file = sys.argv[1]
test_file = sys.argv[2]

matches = pd.read_csv(train_file)
test_matches = pd.read_csv(test_file)



# Cleaning data for non numeric values
matches["team_code_home"] = matches["team_abbreviation_home"].astype("category").cat.codes
matches["team_code_away"] = matches["team_abbreviation_away"].astype("category").cat.codes
matches["season_type_code"] = matches["season_type"].astype("category").cat.codes
#converts last 5 games to a proportional float value of percentage of wins to losses
matches["home_wl_pre5_proportion"] = matches["home_wl_pre5"].apply(lambda x: sum(1 if char == 'W' else 0 for char in x) / 5)
matches["away_wl_pre5_proportion"] = matches["away_wl_pre5"].apply(lambda x: sum(1 if char == 'W' else 0 for char in x) / 5)
matches["home_wl_pre5_num"] = matches["home_wl_pre5"].apply(lambda x: int(''.join('1' if char == 'W' else '0' for char in x)))
matches["away_wl_pre5_num"] = matches["away_wl_pre5"].apply(lambda x: int(''.join('1' if char == 'W' else '0' for char in x)))

#combo stats home+away
matches.loc[:, 'fg_pct_total_avg5'] = matches['fg_pct_home_avg5'] + matches['fg_pct_away_avg5']
matches.loc[:, 'fg3_pct_total_avg5'] = matches['fg3_pct_home_avg5'] + matches['fg3_pct_away_avg5']
matches.loc[:, 'ft_pct_total_avg5'] = matches['ft_pct_home_avg5'] + matches['ft_pct_away_avg5']
matches.loc[:, 'pts_total_avg5'] = matches['pts_home_avg5'] + matches['pts_away_avg5']
matches.loc[:, 'tov_total_avg5'] = matches['tov_home_avg5'] + matches['tov_away_avg5']
matches.loc[:, 'blk_total_avg5'] = matches['blk_home_avg5'] + matches['blk_away_avg5']
matches.loc[:, 'reb_total_avg5'] = matches['reb_home_avg5'] + matches['reb_away_avg5']

matches.loc[:, 'oreb_total_avg5'] = matches['oreb_home_avg5'] + matches['oreb_away_avg5']
matches.loc[:, 'dreb_total_avg5'] = matches['dreb_home_avg5'] + matches['dreb_away_avg5']
matches.loc[:, 'pf_total_avg5'] = matches['pf_home_avg5'] + matches['pf_away_avg5']
matches.loc[:, 'ast_total_avg5'] = matches['ast_home_avg5'] + matches['ast_away_avg5']

#combo stats different features
matches.loc[:, 'ast_reb_home_avg5'] = matches['ast_home_avg5'] + matches['reb_home_avg5']
matches.loc[:, 'ast_reb_away_avg5'] = matches['ast_away_avg5'] + matches['reb_away_avg5']
matches.loc[:, 'pf_tov_home_avg5'] = matches['pf_home_avg5'] + matches['tov_home_avg5']
matches.loc[:, 'pf_tov_away_avg5'] = matches['pf_away_avg5'] + matches['tov_away_avg5']
matches.loc[:, 'pts_fg_pct_home_avg5'] = matches['pts_home_avg5'] + matches['fg_pct_home_avg5']
matches.loc[:, 'pts_fg_pct_away_avg5'] = matches['pts_away_avg5'] + matches['fg_pct_away_avg5']

matches.loc[:, 'tov_ast_home_avg5'] = matches['tov_home_avg5'] + matches['ast_home_avg5']
matches.loc[:, 'tov_ast_away_avg5'] = matches['tov_away_avg5'] + matches['ast_away_avg5']
matches.loc[:, 'ast_fg_pct_home_avg5'] = matches['ast_home_avg5'] + matches['fg_pct_home_avg5']
matches.loc[:, 'ast_fg_pct_away_avg5'] = matches['ast_away_avg5'] + matches['fg_pct_away_avg5']
matches.loc[:, 'fantasy_score_home'] = (
    1.5 * matches['ast_home_avg5'] +
    1.2 * matches['reb_home_avg5'] +
    3 * matches['stl_home_avg5'] +
    3 * matches['blk_home_avg5'] +
    -1 * matches['tov_home_avg5'] +
    matches['pts_home_avg5']
)
matches.loc[:, 'fantasy_score_away'] = (
    1.5 * matches['ast_away_avg5'] +
    1.2 * matches['reb_away_avg5'] +
    3 * matches['stl_away_avg5'] +
    3 * matches['blk_away_avg5'] +
    -1 * matches['tov_away_avg5'] +
    matches['pts_away_avg5']
)
matches.loc[:, 'fantasy_score_diff'] = matches['fantasy_score_home'] - matches['fantasy_score_away']
matches.loc[:, 'stl_fg_pct_home_avg5'] = matches['stl_home_avg5'] + matches['fg_pct_home_avg5']
matches.loc[:, 'stl_fg_pct_away_avg5'] = matches['stl_away_avg5'] + matches['fg_pct_away_avg5']
matches.loc[:, 'stl_fg3_pct_home_avg5'] = matches['stl_home_avg5'] + matches['fg3_pct_home_avg5']
matches.loc[:, 'stl_fg3_pct_away_avg5'] = matches['stl_away_avg5'] + matches['fg3_pct_away_avg5']
matches.loc[:, 'blk_fg_pct_home_avg5'] = matches['blk_home_avg5'] + matches['fg_pct_home_avg5']
matches.loc[:, 'blk_fg_pct_away_avg5'] = matches['blk_away_avg5'] + matches['fg_pct_away_avg5']
matches.loc[:, 'blk_fg3_pct_home_avg5'] = matches['blk_home_avg5'] + matches['fg3_pct_home_avg5']
matches.loc[:, 'blk_fg3_pct_away_avg5'] = matches['blk_away_avg5'] + matches['fg3_pct_away_avg5']
matches.loc[:, 'tov_fg_pct_home_avg5'] = matches['tov_home_avg5'] + matches['fg_pct_home_avg5']
matches.loc[:, 'tov_fg_pct_away_avg5'] = matches['tov_away_avg5'] + matches['fg_pct_away_avg5']
matches.loc[:, 'tov_fg3_pct_home_avg5'] = matches['tov_home_avg5'] + matches['fg3_pct_home_avg5']
matches.loc[:, 'tov_fg3_pct_away_avg5'] = matches['tov_away_avg5'] + matches['fg3_pct_away_avg5']

matches.loc[:, 'reb_fg3_pct_home_avg5'] = matches['reb_home_avg5'] + matches['fg3_pct_home_avg5']
matches.loc[:, 'reb_fg3_pct_away_avg5'] = matches['reb_away_avg5'] + matches['fg3_pct_away_avg5']
matches.loc[:, 'fg_pct_fg3_pct_home_avg5'] = matches['fg_pct_home_avg5'] + matches['fg3_pct_home_avg5']
matches.loc[:, 'fg_pct_fg3_pct_away_avg5'] = matches['fg_pct_away_avg5'] + matches['fg3_pct_away_avg5']
matches.loc[:, 'oreb_fg_pct_home_avg5'] = matches['oreb_home_avg5'] + matches['fg_pct_home_avg5']
matches.loc[:, 'oreb_fg_pct_away_avg5'] = matches['oreb_away_avg5'] + matches['fg_pct_away_avg5']
matches.loc[:, 'oreb_fg3_pct_home_avg5'] = matches['oreb_home_avg5'] + matches['fg3_pct_home_avg5']



matches.loc[:, '3reb_away_reb_home'] = 6.25 * matches['reb_away_avg5'] + 1 * matches['reb_home_avg5']
matches.loc[:, '2blk_away_blk_home'] = 2 * matches['blk_away_avg5'] + 1 * matches['blk_home_avg5']
matches.loc[:, '3stl_away_stl_home'] = 6 * matches['stl_away_avg5'] + 1 * matches['stl_home_avg5']


matches.loc[:, '2oreb_away_ast_away'] = 2 * matches['oreb_away_avg5'] + matches['ast_away_avg5']
matches.loc[:, '2oreb_away_reb_away'] = 2 * matches['oreb_away_avg5'] + 1*matches['reb_away_avg5']




#Clean Test Data
test_matches["team_code_home"] = test_matches["team_abbreviation_home"].astype("category").cat.codes
test_matches["team_code_away"] = test_matches["team_abbreviation_away"].astype("category").cat.codes
test_matches["season_type_code"] = test_matches["season_type"].astype("category").cat.codes

test_matches["home_wl_pre5_proportion"] = test_matches["home_wl_pre5"].apply(
    lambda x: sum(1 if char == 'W' else 0 for char in x) / 5
)
test_matches["away_wl_pre5_proportion"] = test_matches["away_wl_pre5"].apply(
    lambda x: sum(1 if char == 'W' else 0 for char in x) / 5
)
test_matches["home_wl_pre5_num"] = test_matches["home_wl_pre5"].apply(lambda x: int(''.join('1' if char == 'W' else '0' for char in x)))
test_matches["away_wl_pre5_num"] = test_matches["away_wl_pre5"].apply(lambda x: int(''.join('1' if char == 'W' else '0' for char in x)))

#combo stats home+away
test_matches.loc[:, 'fg_pct_total_avg5'] = test_matches['fg_pct_home_avg5'] + test_matches['fg_pct_away_avg5']
test_matches.loc[:, 'fg3_pct_total_avg5'] = test_matches['fg3_pct_home_avg5'] + test_matches['fg3_pct_away_avg5']
test_matches.loc[:, 'ft_pct_total_avg5'] = test_matches['ft_pct_home_avg5'] + test_matches['ft_pct_away_avg5']
test_matches.loc[:, 'pts_total_avg5'] = test_matches['pts_home_avg5'] + test_matches['pts_away_avg5']
test_matches.loc[:, 'tov_total_avg5'] = test_matches['tov_home_avg5'] + test_matches['tov_away_avg5']
test_matches.loc[:, 'blk_total_avg5'] = test_matches['blk_home_avg5'] + test_matches['blk_away_avg5']
test_matches.loc[:, 'reb_total_avg5'] = test_matches['reb_home_avg5'] + test_matches['reb_away_avg5']
test_matches.loc[:, 'oreb_total_avg5'] = test_matches['oreb_home_avg5'] + test_matches['oreb_away_avg5']
test_matches.loc[:, 'dreb_total_avg5'] = test_matches['dreb_home_avg5'] + test_matches['dreb_away_avg5']
test_matches.loc[:, 'pf_total_avg5'] = test_matches['pf_home_avg5'] + test_matches['pf_away_avg5']
test_matches.loc[:, 'ast_total_avg5'] = test_matches['ast_home_avg5'] + test_matches['ast_away_avg5']
#combo stats different features
test_matches.loc[:, 'ast_reb_home_avg5'] = test_matches['ast_home_avg5'] + test_matches['reb_home_avg5']
test_matches.loc[:, 'ast_reb_away_avg5'] = test_matches['ast_away_avg5'] + test_matches['reb_away_avg5']
test_matches.loc[:, 'pf_tov_home_avg5'] = test_matches['pf_home_avg5'] + test_matches['tov_home_avg5']
test_matches.loc[:, 'pf_tov_away_avg5'] = test_matches['pf_away_avg5'] + test_matches['tov_away_avg5']
test_matches.loc[:, 'pts_fg_pct_home_avg5'] = test_matches['pts_home_avg5'] + test_matches['fg_pct_home_avg5']
test_matches.loc[:, 'pts_fg_pct_away_avg5'] = test_matches['pts_away_avg5'] + test_matches['fg_pct_away_avg5']

test_matches.loc[:, 'tov_ast_home_avg5'] = test_matches['tov_home_avg5'] + test_matches['ast_home_avg5']
test_matches.loc[:, 'tov_ast_away_avg5'] = test_matches['tov_away_avg5'] + test_matches['ast_away_avg5']
test_matches.loc[:, 'ast_fg_pct_home_avg5'] = test_matches['ast_home_avg5'] + test_matches['fg_pct_home_avg5']
test_matches.loc[:, 'ast_fg_pct_away_avg5'] = test_matches['ast_away_avg5'] + test_matches['fg_pct_away_avg5']
test_matches.loc[:, 'fantasy_score_home'] = (
    1.5 * test_matches['ast_home_avg5'] +
    1.2 * test_matches['reb_home_avg5'] +
    3 * test_matches['stl_home_avg5'] +
    3 * test_matches['blk_home_avg5'] +
    -1 * test_matches['tov_home_avg5'] +
    test_matches['pts_home_avg5']
)
test_matches.loc[:, 'fantasy_score_away'] = (
    1.5 * test_matches['ast_away_avg5'] +
    1.2 * test_matches['reb_away_avg5'] +
    3 * test_matches['stl_away_avg5'] +
    3 * test_matches['blk_away_avg5'] +
    -1 * test_matches['tov_away_avg5'] +
    test_matches['pts_away_avg5']
)
test_matches.loc[:, 'fantasy_score_diff'] = test_matches['fantasy_score_home'] - test_matches['fantasy_score_away']
test_matches.loc[:, 'stl_fg_pct_home_avg5'] = test_matches['stl_home_avg5'] + test_matches['fg_pct_home_avg5']
test_matches.loc[:, 'stl_fg_pct_away_avg5'] = test_matches['stl_away_avg5'] + test_matches['fg_pct_away_avg5']
test_matches.loc[:, 'stl_fg3_pct_home_avg5'] = test_matches['stl_home_avg5'] + test_matches['fg3_pct_home_avg5']
test_matches.loc[:, 'stl_fg3_pct_away_avg5'] = test_matches['stl_away_avg5'] + test_matches['fg3_pct_away_avg5']
test_matches.loc[:, 'blk_fg_pct_home_avg5'] = test_matches['blk_home_avg5'] + test_matches['fg_pct_home_avg5']
test_matches.loc[:, 'blk_fg_pct_away_avg5'] = test_matches['blk_away_avg5'] + test_matches['fg_pct_away_avg5']
test_matches.loc[:, 'blk_fg3_pct_home_avg5'] = test_matches['blk_home_avg5'] + test_matches['fg3_pct_home_avg5']
test_matches.loc[:, 'blk_fg3_pct_away_avg5'] = test_matches['blk_away_avg5'] + test_matches['fg3_pct_away_avg5']
test_matches.loc[:, 'tov_fg_pct_home_avg5'] = test_matches['tov_home_avg5'] + test_matches['fg_pct_home_avg5']
test_matches.loc[:, 'tov_fg_pct_away_avg5'] = test_matches['tov_away_avg5'] + test_matches['fg_pct_away_avg5']
test_matches.loc[:, 'tov_fg3_pct_home_avg5'] = test_matches['tov_home_avg5'] + test_matches['fg3_pct_home_avg5']
test_matches.loc[:, 'tov_fg3_pct_away_avg5'] = test_matches['tov_away_avg5'] + test_matches['fg3_pct_away_avg5']
test_matches.loc[:, 'reb_fg3_pct_home_avg5'] = test_matches['reb_home_avg5'] + test_matches['fg3_pct_home_avg5']
test_matches.loc[:, 'reb_fg3_pct_away_avg5'] = test_matches['reb_away_avg5'] + test_matches['fg3_pct_away_avg5']
test_matches.loc[:, 'fg_pct_fg3_pct_home_avg5'] = test_matches['fg_pct_home_avg5'] + test_matches['fg3_pct_home_avg5']
test_matches.loc[:, 'fg_pct_fg3_pct_away_avg5'] = test_matches['fg_pct_away_avg5'] + test_matches['fg3_pct_away_avg5']
test_matches.loc[:, 'oreb_fg_pct_home_avg5'] = test_matches['oreb_home_avg5'] + test_matches['fg_pct_home_avg5']
test_matches.loc[:, 'oreb_fg_pct_away_avg5'] = test_matches['oreb_away_avg5'] + test_matches['fg_pct_away_avg5']
test_matches.loc[:, 'oreb_fg3_pct_home_avg5'] = test_matches['oreb_home_avg5'] + test_matches['fg3_pct_home_avg5']



test_matches.loc[:, '3reb_away_reb_home'] = 6.25 * test_matches['reb_away_avg5'] + 1 * test_matches['reb_home_avg5']
test_matches.loc[:, '2blk_away_blk_home'] = 2 * test_matches['blk_away_avg5'] + test_matches['blk_home_avg5']
test_matches.loc[:, '3stl_away_stl_home'] = 6 * test_matches['stl_away_avg5'] + 1 * test_matches['stl_home_avg5']


test_matches.loc[:, '2oreb_away_ast_away'] = 2 * test_matches['oreb_away_avg5'] + test_matches['ast_away_avg5']
test_matches.loc[:, '2oreb_away_reb_away'] = 2 * test_matches['oreb_away_avg5'] + 1*test_matches['reb_away_avg5']










target = matches["label"]
train_data = matches.drop(columns=["label"])
train_data = train_data.drop(columns=["team_abbreviation_home"])
train_data = train_data.drop(columns=["team_abbreviation_away"])
train_data = train_data.drop(columns=["season_type"])
train_data = train_data.drop(columns=["home_wl_pre5"])
train_data = train_data.drop(columns=["away_wl_pre5"])


priors = target.value_counts(normalize=True).to_dict()
# print(class_priors)

predictors = []
# predictors.append("team_code_home")
# predictors.append("team_code_away")
# predictors.append("season_type_code")
# predictors.append("home_wl_pre5_proportion")
predictors.append("away_wl_pre5_proportion")
predictors.append("min_avg5") # important
predictors.append("fg_pct_home_avg5")
predictors.append("fg3_pct_home_avg5") #loses 1 test
predictors.append("ft_pct_home_avg5")
# predictors.append("oreb_home_avg5") #important
predictors.append("dreb_home_avg5")
predictors.append("reb_home_avg5")
# predictors.append("ast_home_avg5") #loses 2 cases
predictors.append("stl_home_avg5")
predictors.append("blk_home_avg5")
predictors.append("tov_home_avg5") #important
predictors.append("pf_home_avg5")
# predictors.append("pts_home_avg5") #lowered by 1 test
predictors.append("fg_pct_away_avg5")
# predictors.append("fg3_pct_away_avg5")
predictors.append("ft_pct_away_avg5") #maybe
# predictors.append("oreb_away_avg5") #maybe
predictors.append("dreb_away_avg5") #important
# predictors.append("reb_away_avg5") #important #maybe #maybe
predictors.append("ast_away_avg5")
predictors.append("stl_away_avg5")
# predictors.append("blk_away_avg5") #maybe
predictors.append("tov_away_avg5")
predictors.append("pf_away_avg5") #important
predictors.append("pts_away_avg5") #important
# predictors.append("fg_pct_total_avg5")
# predictors.append("fg3_pct_total_avg5")
predictors.append("ft_pct_total_avg5") #lowered by 1 test
# predictors.append("pts_total_avg5") #maybe
# predictors.append("tov_total_avg5") #maybe   #
# predictors.append("blk_total_avg5")
# predictors.append("reb_total_avg5")
# predictors.append("oreb_total_avg5") #maybe   #
predictors.append("dreb_total_avg5")
# predictors.append("pf_total_avg5") #maybe   #
# predictors.append("ast_total_avg5")
# predictors.append("ast_reb_home_avg5")
predictors.append("ast_reb_away_avg5")
# predictors.append("pf_tov_home_avg5")
# predictors.append("pf_tov_away_avg5")
# predictors.append("pts_fg_pct_home_avg5")
predictors.append("pts_fg_pct_away_avg5")

# predictors.append("home_wl_pre5_num")
# predictors.append("away_wl_pre5_num")  #maybe #maybe #lose 1 case

# predictors.append("tov_ast_home_avg5")
predictors.append("tov_ast_away_avg5") #maybe
# predictors.append("ast_fg_pct_home_avg5")
predictors.append("ast_fg_pct_away_avg5")
# predictors.append("fantasy_score_home")
# predictors.append("fantasy_score_away")
# predictors.append("fantasy_score_diff")
# predictors.append("stl_fg_pct_home_avg5")
predictors.append("stl_fg_pct_away_avg5")
predictors.append("stl_fg3_pct_home_avg5")
# predictors.append("stl_fg3_pct_away_avg5") #maybe
# predictors.append("blk_fg_pct_home_avg5")
predictors.append("blk_fg_pct_away_avg5")
# predictors.append("blk_fg3_pct_home_avg5")
# predictors.append("blk_fg3_pct_away_avg5") #maybe #maybe 0.08
# predictors.append("tov_fg_pct_home_avg5")
predictors.append("tov_fg_pct_away_avg5")   
# predictors.append("tov_fg3_pct_home_avg5")
# predictors.append("tov_fg3_pct_away_avg5")


# predictors.append("reb_fg3_pct_home_avg5")
# predictors.append("reb_fg3_pct_away_avg5") #maybe

# predictors.append("fg_pct_fg3_pct_home_avg5")
# predictors.append("fg_pct_fg3_pct_away_avg5")
predictors.append("oreb_fg_pct_home_avg5") #maybe
# predictors.append("oreb_fg_pct_away_avg5")
predictors.append("oreb_fg3_pct_home_avg5") #maybe


predictors.append("3reb_away_reb_home")#maybe
# predictors.append("2blk_away_blk_home")
predictors.append("3stl_away_stl_home")

# predictors.append("2oreb_away_ast_away") #maybe .06
predictors.append("2oreb_away_reb_away")





#Actuall Probability Calculation
train_data = train_data[predictors]
test_data = test_matches[predictors]


means = {}
variances = {}


for outcome in priors:
    # print("OUTCOME", outcome)
    predictor = train_data[target == outcome]
    # print("PREDICTOR", predictor.head())
    predictor_mean = predictor.mean()
    predictor_var = predictor.var()
    means[outcome] = predictor_mean
    variances[outcome] = predictor_var
    priors[outcome] = len(predictor) / len(train_data)

# print(means)
# print(variances)

def gaussian(x, mean, variance):
    return (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-((x - mean) ** 2) / (2 * variance))

def predict(row):
    probabilities = {}
    for outcome in priors:
        probabilities[outcome] = priors[outcome]
        for i in range(len(row)):
            probabilities[outcome] *= gaussian(row[i], means[outcome].iloc[i], variances[outcome].iloc[i])
    return max(probabilities, key=probabilities.get)

def predict_log(row):
    log_probabilities = {}
    for outcome in priors:
        log_probabilities[outcome] = np.log(priors[outcome])
        for i in range(len(row)):
            mean = means[outcome].iloc[i]
            variance = variances[outcome].iloc[i]
            log_probabilities[outcome] += gaussian_log(row[i], mean, variance)
    return max(log_probabilities, key=log_probabilities.get)

def predict_all(data):
    return [predict_log(row) for row in data.values]


def gaussian_log(x, mean, variance):
    return -0.5 * np.log(2 * np.pi * variance) - ((x - mean) ** 2) / (2 * variance)


def accuracy(predictions, actual):
    predictions = np.array(predictions)
    actual = actual.values
    # print("PREDICTIONS", predictions)
    # print("ACTUAL", actual)
    return np.sum(predictions == actual) / len(actual)

test_predictions = predict_all(test_data)

for prediction in test_predictions:
    print(prediction)

# print(accuracy(test_predictions, test_matches["label"]))


# print(means)
# print(variances)
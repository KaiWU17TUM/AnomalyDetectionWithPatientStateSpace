selected_pharma_file = '../processed/HiRID_selected_variables-input_dict.csv'
selected_pharma = pd.read_csv(selected_pharma_file)

# duplicated variables to be merged or added up
merge_items = {
    'merge': [
        [300, 310],
        [4000, 8280],
        [24000835, 24000866],
        [30005010, 30005110],
    ],
    'add': [
        [10010020, 10010070, 10010071, 10010072],
    ],
}

selected_physio = pd.read_csv('../processed/HiRID_selected_variables-output_processed.csv')

uid_dict = {}
for index, row in selected_physio.iterrows():
    uid_dict[row['variableid']] = row['uid']
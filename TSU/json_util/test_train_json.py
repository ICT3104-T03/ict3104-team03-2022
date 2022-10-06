import json
import os.path
from sklearn.utils import shuffle


def create_subset_json(no_of_test, no_of_train, input_file_name, input_file_directory, output_file_name, output_file_directory):
    """ Creates an output json with entries that is a subset of the input json

    Arguments:
    no_of_test -- number of test entries to extract to json subset
    no_of_train -- number of training entries to extract to json subset
    input_file_name -- input json file to extract the data from
    input_file_directory -- relative file path to directory containing input_file_name
    output_file_name -- name of new output json file
    output_file_directory -- relative file path to directory that will contain output_file_name

    Example:

    -Import/setup-
    # absolute path to the json_util package from where this code/OS is run
    # import test_train_json.py and the functions from inside it
    from TSU.json_util import test_train_json 

    -Usage-
    no_of_test = 2
    no_of_train = 0
    input_file_name = "smarthome_CS_51_old"
    input_file_directory = "./TSU/tsu_data/"
    output_file_name = "hello"
    output_file_directory = "./TSU/tsu_data/"
    test_train_json.create_subset_json(no_of_test, no_of_train, input_file_name, input_file_directory, output_file_name, output_file_directory)
    """
    file = input_file_directory

    # user input both testing and training
    user_selecttesting = no_of_test
    user_selecttraining = no_of_train
    user_pathfilesave = output_file_directory
    file += input_file_name + ".json"
    # open user selected file
    f = open(file)
    data = json.load(f)
    testing = []
    training = []
    final_training = []
    final_testing = []
    # Run file to get data name in new array where it is testing or training
    for info in data:
        if data[info]['subset'] == "testing":
            testing.append(info)
        else:
            training.append(info)
    # randomizing the order of array
    testing = shuffle(testing)
    training = shuffle(training)
    # saving the whole data into array for both training and testing
    for i in range(user_selecttesting):
        for datatest in data:
            if datatest == testing[i]:
                result1 = (datatest, data[datatest])
                final_testing.append(result1)
    for i in range(user_selecttraining):
        for datatest in data:
            if datatest == training[i]:
                result = (datatest, data[datatest])
                final_training.append(result)
    # save into a new json file
    with open(user_pathfilesave+output_file_name+'.json', mode='w+') as f:
        json.dump(final_testing, f, indent=2)
        json.dump(final_training, f, indent=2)

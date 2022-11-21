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
    input_file_name = "smarthome_CS_51.json"
    input_file_directory = "./TSU/tsu_data/"
    output_file_name = "train_smarthome_CS.json"
    output_file_directory = "./TSU/tsu_data/"
    test_train_json.create_subset_json(no_of_test, no_of_train, input_file_name, input_file_directory, output_file_name, output_file_directory)
    """
    file = input_file_directory

    # user input both testing and training
    user_selecttesting = no_of_test
    user_selecttraining = no_of_train
    user_pathfilesave = output_file_directory
    file += input_file_name
    # open user selected file
    f = open(file)
    data = json.load(f)
    testing = []
    training = []
    final_json = {}
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
                final_json[datatest] = data[datatest]
    for i in range(user_selecttraining):
        for datatest in data:
            if datatest == training[i]:
                final_json[datatest] = data[datatest]
    # save into a new json file
    with open(user_pathfilesave+output_file_name, mode='w+') as f:
        json.dump(final_json, f, indent=2)

    return len(final_json)

'''
use to count number of test and train in the file
'''
def count_train_test(file_name, dataset_type):
    file = None
    training = 0
    testing = 0
    if dataset_type == "TSU":
        file = "./TSU/tsu_data/"+str(file_name)
        with open(file, mode='r') as f:
            data = json.load(f)
        
    for info in data:
        if data[info]['subset'] == "training":
            training += 1
        if data[info]['subset'] == "testing":
            testing += 1
    
    return training, testing

'''
use to load just 1 video data into inferencing_dataset.json for inferencing
'''
def inference_json(video_name, dataset_type):
    file = None
    final_json = {}
    if dataset_type == "TSU":
        file = "./TSU/tsu_data/smarthome_CS_51.json"
        with open(file, mode='r') as f:
            data = json.load(f)
        video = video_name.replace(".mp4", "")
        for info in data:
            if info == video:
                final_json[info] = data[info]
        with open("./TSU/tsu_data/inference_smarthome_CS_51.json", mode='w+') as f:
            json.dump(final_json, f, indent = 2)
    return video
import glob
import pickle
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from federatedlearner import NNWorker
import data.federated_data_extractor as dataext

# Function to load the most recent global model from the blocks folder
def get_global_model_from_blocks_folder():
    # Get a list of block files
    block_files = glob.glob('blocks/*.block')
    # Sort the files by their names (assuming higher names correspond to more recent blocks)
    block_files.sort()

    # Load the most recent block
    if block_files:
        with open(block_files[-1], 'rb') as f:
            block = pickle.load(f)
        return block.accuracy, block.index, block.updates
    else:
        print("No blocks found in the blocks folder.")
        return None

# Load the test dataset
def load_test_dataset():
    dataset = dataext.load_data("data/sa.d")
    return dataset['test_text'], dataset['test_labels']

# Evaluate the global model
def evaluate_global_model():
    X_test, Y_test = load_test_dataset()  # Load your test dataset
    acc, id, upd = get_global_model_from_blocks_folder()

    # if global_model is not None:
    #     # Initialize NNWorker
    #     worker = NNWorker(tX=X_test, tY=Y_test)
    #     worker.build(global_model)  # Build network with the global model
    #
    #     # Evaluate the model on the test data2
    #
    #     global_model_accuracy = worker.evaluate()


        # Close the session
    #     worker.close()
    # else:
    #     print("Failed to load the global model.")
    print("Round: ", id, "Global Model Accuracy:", acc)
    for client in upd.keys():
        print(upd[client].computing_time)




if __name__ == '__main__':
    evaluate_global_model()

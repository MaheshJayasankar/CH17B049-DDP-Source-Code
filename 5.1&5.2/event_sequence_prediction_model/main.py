import pickle
import os
from event_gen_code import NewRunTest
import warnings
warnings.filterwarnings("ignore")

# Written for use in Dual Degree Project, 2021-2022 by J Mahesh, CH17B049, Dept. Chemical Engineering, IIT Madras
# An adaptive program to run the Event Sequence Prediction Model vs Random Enumeration Method analysis over several batches.
# The program can be interrupted and will resume from the latest incomplete batch.
# Order of execution: testbatcha -> batcha (1 - 10) -> testbatchb -> batchb (1 - 10)

# The results are stored as .pkl files

# After execution, the results_plot.py and results_analysis.py may be called to obtain the output

MAX_BATCHES = 10
AMT_PER_BATCH = 2

def run_batch(filename, seed_list, iter_count, random_only = False):

    # This function initiates the main algorithm from event_gen_code.py
    res_dict = NewRunTest(seed_list,iter_count, random_only)

    print("Job completed. Saving file:")

    with open(filename, 'wb') as handle:
        pickle.dump(res_dict, handle)

    print(f"File saved as {filename}. Verifying:")

    with open(filename, 'rb') as handle:
        load_dict = pickle.load(handle)

    print(f"Load operation verified: {load_dict == res_dict}")
    print(f"Loaded file:")
    print(f"{str(load_dict)[:50]} ...")

def search_test_batches():
    test_batches = ['testbatcha.pkl', 'testbatchb.pkl']
    print("Analyzing file directory...")
    if os.path.exists(test_batches[0]):
        if os.path.exists(test_batches[1]):
            print("Test batches were completed.")
            return 2
        else:
            print("Test batch a is completed, but test batch b is not.")
            return 1
    print("Test batches not completed. Starting test batches...")
    return 0

def search_batch(batchprefix):
    idx = 1
    while (os.path.exists(f"{batchprefix}{idx}.pkl")):
        idx += 1
    if idx <= MAX_BATCHES:
        print(f"{batchprefix} not completed fully. Resuming from {batchprefix}{idx}")
        return idx
    else:
        return -1

def search_all_batches():
    test_status = search_test_batches()
    if test_status == 0:
        main_loop(0)
        return
    elif test_status == 1:
        batch_progress = search_batch("batcha")
        if (batch_progress < 0):
            main_loop(2)
            return
        else:
            main_loop(1, batch_progress)
            return
    elif test_status == 2:
        batch_progress = search_batch("batchb")
        if (batch_progress < 0):
            main_loop(4)
            return
        else:
            main_loop(3, batch_progress)
            return
        


def main_loop(start_pos, batch_pos = 1):

    test_itr_count = 5
    batch_itr_count = 500

    if start_pos <= 0:
        print("\nRunning Test Batch A\n")
        run_batch("testbatcha.pkl", [0,1], test_itr_count)
        print("Test Batch A Completed\n")
    if start_pos <= 1:
        while(batch_pos <= MAX_BATCHES):
            print(f"\nRunning Batch A{batch_pos}\n")
            run_batch(f"batcha{batch_pos}.pkl", range((batch_pos - 1) * AMT_PER_BATCH, batch_pos * AMT_PER_BATCH), batch_itr_count)
            batch_pos += 1
        batch_pos = 1
        print("Batch A all completed.\n")
    if start_pos <= 2:
        print("\nRunning Test Batch B\n")
        run_batch("testbatchb.pkl", [0,1], test_itr_count, True)
        print("Test Batch B Completed\n")
    if start_pos <= 3:
        while(batch_pos <= MAX_BATCHES):
            print(f"\nRunning Batch B{batch_pos}\n")
            run_batch(f"batchb{batch_pos}.pkl", range((batch_pos - 1) * AMT_PER_BATCH, batch_pos * AMT_PER_BATCH), batch_itr_count, True)
            batch_pos += 1
        print("Batch B all completed.\n")
    if start_pos <= 4:
        print("All batches completed.")

search_all_batches()
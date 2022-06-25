import pickle
from event_gen_code import NewRunTest

filename = "testbatchb.pkl"
seed_list = [0,1]
iter_count = 3

res_dict = NewRunTest(seed_list,iter_count, modify = True)

print("Job completed. Saving file:")

with open(filename, 'wb') as handle:
    pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"File saved as {filename}. Verifying:")

with open(filename, 'rb') as handle:
    load_dict = pickle.load(handle)

print(f"Load operation verified: {load_dict == res_dict}")
print(f"Loaded file:")
print(f"{str(load_dict)[:50]} ...")
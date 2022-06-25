import pickle
from event_gen_code import NewRunTest

filename = "batcha1.pkl"
seed_list = [0,1,2,3,4]
iter_count = 500

res_dict = NewRunTest(seed_list,iter_count)

print("Job completed. Saving file:")

with open(filename, 'wb') as handle:
    pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"File saved as {filename}. Verifying:")

with open(filename, 'rb') as handle:
    load_dict = pickle.load(handle)

print(f"Load operation verified: {load_dict == res_dict}")
print(f"Loaded file:")
print(f"{str(load_dict)[:50]} ...")
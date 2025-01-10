
def best_combination(list_of_dict, sort_key):
    # Check if the sort_key exists in the dictionaries
    if not all(isinstance(item, dict) and sort_key in item for item in list_of_dict):
        raise ValueError(f"All dictionaries must contain the key '{sort_key}'.")

    # Sort the list of dictionaries based on the sort_key
    sorted_data = sorted(list_of_dict, key=lambda x: x[sort_key])
    return sorted_data

def best_combination_keys(list_of_dict, sort_keys):
    
    # Validate that all dictionaries have the specified sort_keys
    for sort_key in sort_keys:
        if not all(sort_key in d for d in list_of_dict):
            raise ValueError(f"All dictionaries must contain the key '{sort_key}'.")
    
    # Validate that the sort_keys correspond to numerical values
    for sort_key in sort_keys:
        if not all(isinstance(d[sort_key], (int, float)) for d in list_of_dict):
            raise ValueError(f"The key '{sort_key}' must correspond to numerical values in all dictionaries.")

    # Sort the list of dictionaries
    # return sorted(input_list, key=lambda x: x[sort_keys])
    return sorted(list_of_dict, key=lambda x: tuple(x[key] for key in sort_keys))

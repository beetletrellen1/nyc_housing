def home_size_mask(size_value):
    if size_value == "Large Home": size_mask = [1]
    elif size_value == "Small Home": size_mask = [0]
    else: size_mask = [0,1]
    return size_mask

def home_age_mask(age_value):
    if age_value == "New Home": age_mask = [1]
    elif age_value == "Old Home": age_mask = [0]
    else: age_mask = [0,1]
    return age_mask
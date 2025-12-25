def drug_profile(profile):
    if profile == "Headache":
        return {"MW": (150, 400), "logP": (1, 3)}
    if profile == "Antibiotic":
        return {"MW": (200, 600), "logP": (-1, 3)}
    if profile == "Anti-inflammatory":
        return {"MW": (200, 500), "logP": (2, 5)}
    return {}

def passes_filter(props, profile):
    rules = drug_profile(profile)
    for key, (low, high) in rules.items():
        if not (low <= props[key] <= high):
            return False
    return True

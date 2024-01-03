from pymedtermino.snomedct import SNOMEDCT
#from pymedtermino.snomedct import build
#build.main()
def standardize_medical_keyword_ontology(keyword):
    # Use SNOMED CT to find the preferred term
    concept = SNOMEDCT.search(keyword)

    if concept:
        # Return the preferred term from SNOMED CT
        return concept[0].preferred
    else:
        # If not found, return the original keyword
        return keyword

# Example list of medical keywords
medical_keywords = ["Heart Disease", "cancer - malignant", "Diabetes Mellitus", "  Vaccination", "Surgical Procedure","urine"]

# Standardize each keyword using SNOMED CT ontology
standardized_keywords_ontology = [standardize_medical_keyword_ontology(keyword) for keyword in medical_keywords]

# Display the results
for original, standardized in zip(medical_keywords, standardized_keywords_ontology):
    print(f"Original: {original}\t Standardized: {standardized}")
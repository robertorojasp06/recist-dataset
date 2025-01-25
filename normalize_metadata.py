import pandas as pd
import argparse
import unicodedata
from pathlib import Path


def normalize_spanish_string(string):
    # Remove accents and other spanish characters
    nfkd_form = unicodedata.normalize('NFKD', string)
    unicode_string = ''.join([c for c in nfkd_form if not unicodedata.combining(c)]).lower()
    # Remove extra whitespaces
    normalized = ' '.join([word.strip() for word in unicode_string.split()])
    return normalized


class PatientsNormalizer:
    def __init__(self):
        self.diagnoses_mapping = {
            "melanoma": "melanoma",
            "cancer mama": "breast cancer",
            "cancer vejiga": "bladder cancer",
            "cancer colon": "colon cancer",
            "cancer gastrico": "gastric cancer",
            "cancer recto": "rectal cancer",
            "cancer vesicula biliar": "gallbladder cancer",
            "cancer ovario": "ovarian cancer",
            "cancer pulmon": "lung cancer",
            "cancer higado": "liver cancer",
            "cancer pancreatobiliar": "pancreatobiliary cancer"
        }

    def _translate_diagnosis(self, diagnosis):
        translated = self.diagnoses_mapping.get(diagnosis, None)
        if not translated:
            raise ValueError(f"diagnosis '{diagnosis}' is not in the mapping.")
        return translated

    def normalize_patients(self, path_to_patients):
        patients_df = pd.read_csv(path_to_patients)
        # Rename columns
        patients_df.rename(
            columns={
                "protocolo": "protocol",
                "sexo": "sex",
                "edad": "age",
                "diagnóstico": "diagnosis"
            },
            inplace=True
        )
        # Translate diagnoses to english
        patients_df["diagnosis"] = patients_df["diagnosis"].apply(normalize_spanish_string)
        patients_df["diagnosis"] = patients_df["diagnosis"].apply(self._translate_diagnosis)
        return patients_df


def main():
    parser = argparse.ArgumentParser(
        description="""Normalize patients.csv file. Especifically,
        translate diagnoses from spanish to english.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "path_to_patients",
        type=str,
        help="Path to the patients.csv file."
    )
    parser.add_argument(
        "--path_to_output",
        type=str,
        default=Path.cwd(),
        help="Path to the directory to save the output file."
    )
    args = parser.parse_args()
    patients_normalizer = PatientsNormalizer()
    patients_df = patients_normalizer.normalize_patients(args.path_to_patients)
    patients_df.to_csv(
        args.path_to_output / "patients.csv",
        index=False
    )


if __name__ == "__main__":
    main()

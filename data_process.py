import numpy as np
import pandas as pd

# Read the data files.
transpro_df = pd.read_excel(r"...product.xlsx")

# Function to filter spectra based on precursor m/z difference
def filter_spectra(df, tolerance=0.1):
    unique_mz = set()
    filtered_spectra = []
    for _, row in df.iterrows():
        mz = row["mz"]
        is_unique = True
        for existing_mz in unique_mz:
            if abs(mz - existing_mz) <= tolerance:
                is_unique = False
                break
        if is_unique:
            unique_mz.add(mz)
            filtered_spectra.append({
                "id": row["ID"],
                "precursor_mz": mz,
                "peaks": row["MS2"],
                "type": df.columns.tolist()[0].split("_")[0]
            })
    return filtered_spectra


# main
if __name__ == "__main__":
    # Process the data.
    print("Process the data...")
    filtered_spectra = filter_spectra(transpro_df)

    # Convert the processed data into a DataFrame.
    result_data = []
    for spectrum in filtered_spectra:
        result_data.append({
            "ID": spectrum["id"],
            "Precursor_mz": spectrum["precursor_mz"],
            "Type": spectrum["type"],
            "MS2": spectrum["peaks"]  
        })

    result_df = pd.DataFrame(result_data)

    # Save the results to a new Excel file.
    output_file = r"...product_process.xlsx"
    result_df.to_excel(output_file, index=False)
    print(f"Processing completed！results saved to {output_file}")

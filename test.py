import os
import pandas as pd
import json
from src.ei_estimation import ei_estimation_pipeline   
from src.vbr_metric import calculate_brain_metrics
import csv
from dotenv import load_dotenv
from src.pipeline_for_dcm_folder import run_pipeline_dcm_to_data_folder

# Multiple cases pipeline
def run_pipeline_for_multiple_cases(parent_folder, project_root=".", data_folder_name="data", csv_output_path="cases_paths.csv"):
    """
    Runs the DICOM-to-data-folder pipeline for multiple cases in a parent folder.

    Parameters:
        parent_folder (str): Path containing multiple case folders with DICOM files
        project_root (str): Root path of the project
        data_folder_name (str): Folder name for outputs inside the project root
        csv_output_path (str): CSV file path to store results

    Returns:
        None
    """
    load_dotenv()
    segmentator_token = os.getenv("TOTALSEGMENTATOR_TOKEN")
    
    # Prepare CSV
    with open(csv_output_path, mode='w', newline='') as csvfile:
        fieldnames = ["case_folder", "output_dir", "ct_path", "brain_mask_path", "ref_path", "flirt_matrix_path"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Перебираем все папки в parent_folder
        for case_name in os.listdir(parent_folder):
            case_path = os.path.join(parent_folder, case_name)
            if os.path.isdir(case_path):
                print(f"[INFO] Processing case: {case_name}")
                try:
                    paths = run_pipeline_dcm_to_data_folder(
                        dcm_folder_path=case_path,
                        totalseg_token=segmentator_token,
                        project_root=project_root,
                        data_folder_name=data_folder_name
                    )
                    # Добавляем case_folder и нужные пути в CSV
                    row = {
                        "case_folder": case_name,
                        "output_dir": paths.get("output_dir"),
                        "ct_path": paths.get("ct_path"),
                        "brain_mask_path": paths.get("brain_mask_path"),
                        "ref_path": paths.get("ref_path"),
                        "flirt_matrix_path": paths.get("flirt_matrix_path")
                    }
                    writer.writerow(row)
                except Exception as e:
                    print(f"[ERROR] Failed to process {case_name}: {e}")

    print(f"[INFO] All cases processed. CSV saved to {csv_output_path}")

def add_ei_and_metrics_to_csv(csv_input_path, csv_output_path=None):
    """
    Reads CSV with case paths, runs EI estimation and brain metrics,
    and adds results as new columns (EI, VBR, note).
    """
    if csv_output_path is None:
        csv_output_path = csv_input_path  # перезаписываем исходный файл

    # Загружаем CSV
    df = pd.read_csv(csv_input_path)

    # Проверяем, что есть колонка с путём
    if "output_dir" not in df.columns:
        raise ValueError("CSV file must contain 'output_dir' column")

    ei_results, vbr_results, note_results = [], [], []

    for idx, row in df.iterrows():
        base = row["output_dir"]
        case_name = row.get("case_folder", f"case_{idx}")
        print(f"[INFO] Processing case: {case_name}")

        # ---- EI ----
        try:
            results = ei_estimation_pipeline(base)
            ei_value = results.get("hydrocephalus_result", None)
        except Exception as e:
            print(f"[ERROR] EI estimation failed for {case_name}: {e}")
            ei_value = None
        ei_results.append(ei_value)

        # ---- VBR + note ----
        try:
            statistics_path = os.path.join(base, "brain_structures", "statistics.json")
            if os.path.exists(statistics_path):
                metrics_results = calculate_brain_metrics(statistics_path)
                vbr_value = metrics_results.get("VBR", None)
                note_value = metrics_results.get("note", None)
            else:
                vbr_value, note_value = None, None
        except Exception as e:
            print(f"[ERROR] Brain metrics failed for {case_name}: {e}")
            vbr_value, note_value = None, None

        vbr_results.append(vbr_value)
        note_results.append(note_value)

    # Добавляем новые колонки
    df["EI"] = ei_results
    df["VBR"] = vbr_results
    df["note"] = note_results

    # Сохраняем обратно
    df.to_csv(csv_output_path, index=False)
    print(f"[INFO] Updated CSV saved to: {csv_output_path}")

    return df  # сразу возвращаем как DataFrame


if __name__ == "__main__":
    parent_folder = "/Users/maratorozaliev/Desktop/NPH patients"  # путь к папке с кейсами
    csv_cases = "/Users/maratorozaliev/Desktop/EI and VBR sample data/cases_paths_12_run.csv"
    csv_with_metrics = "/Users/maratorozaliev/Desktop/EI and VBR sample data/cases_paths_with_metrics_12_run.csv"

    # --- Шаг 1: обработка кейсов и сохранение путей ---
    run_pipeline_for_multiple_cases(
        parent_folder=parent_folder,
        project_root="/Users/maratorozaliev/Desktop/MindScope",
        data_folder_name="data",
        csv_output_path=csv_cases
    )

    # --- Шаг 2: добавление EI и метрик ---
    df = add_ei_and_metrics_to_csv(
        csv_input_path=csv_cases,
        csv_output_path=csv_with_metrics
    )

    print(df.head())

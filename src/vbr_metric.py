import os
import json

def load_brain_data(json_path):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"File not found: {json_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data

def get_volume(data, key):
    return data.get(key, {}).get("volume", 0)

def calculate_brain_metrics(json_path):
    data = load_brain_data(json_path)

    ventricle_volume = get_volume(data, "ventricle")

    lobar_regions = ["frontal_lobe", "parietal_lobe", "occipital_lobe", "temporal_lobe"]
    if get_volume(data, "cerebellum") > 0:
        lobar_regions.append("cerebellum")

    brain_volume = sum(get_volume(data, region) for region in lobar_regions)

    # Calculate Ventricle-to-Brain-Ratio
    vbr = ventricle_volume / brain_volume if brain_volume > 0 else 0

    return {
        "brain_volume": brain_volume,
        "ventricle_volume": ventricle_volume,
        "VBR": round(vbr, 4),
        "note": ">0.03 may indicate hydrocephalus" if vbr > 0.03 else "within normal range"
    }

if __name__ == "__main__":
    statistics_path = "/Users/maratorozaliev/Desktop/MindScope/data/000518240B_155448/brain_structures/statistics.json"
    metrics_results = calculate_brain_metrics(statistics_path)
    print({key: metrics_results[key] for key in ['VBR', 'note']})
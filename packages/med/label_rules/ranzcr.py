"""RANZCR CLiP: one 0/1 column per device and position. Each image was annotated for the four
devices (Tang et al., 2021), so a device with none of its columns at 1 is absent: negative.

The position (normal, borderline, abnormal, incompletely imaged) does not matter here: the
question is whether the device is there.
"""
DEVICES = {
    "endotracheal_tube": ["ETT - Abnormal", "ETT - Borderline", "ETT - Normal"],
    "nasogastric_tube": ["NGT - Abnormal", "NGT - Borderline", "NGT - Incompletely Imaged", "NGT - Normal"],
    "central_venous_catheter": ["CVC - Abnormal", "CVC - Borderline", "CVC - Normal"],
    "swan_ganz_catheter": ["Swan Ganz Catheter Present"],
}
COLUMNS = [column for columns in DEVICES.values() for column in columns]


def labels(row: dict) -> dict[str, str]:
    return {device: "positive" if any(str(row[column]) == "1" for column in columns) else "negative"
            for device, columns in DEVICES.items()}

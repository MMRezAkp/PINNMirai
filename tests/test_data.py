import pandas as pd
from mirai_pinn.data import CAN_COLUMNS, load_target_excel

def test_excel_adapter_standardises_required_columns(tmp_path):
    raw = pd.DataFrame({name: [0.0, 0.1] for name in CAN_COLUMNS.values()})
    raw[CAN_COLUMNS["soc_pct"]] = [50.0, 51.0]
    path = tmp_path / "input.xlsx"; raw.to_excel(path, index=False)
    data = load_target_excel(path)
    assert {"soc", "dt_s", "temperature_k"}.issubset(data.columns)
    assert data["soc"].tolist() == [0.5, 0.51]

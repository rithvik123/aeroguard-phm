"""Standard NASA C-MAPSS column definitions used by AeroGuard."""

UNIT_COLUMN = "unit_id"
CYCLE_COLUMN = "cycle"

OPERATIONAL_SETTING_COLUMNS = [
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
]

SENSOR_COLUMNS = [f"sensor_{idx}" for idx in range(1, 22)]

CMAPSS_COLUMNS = [
    UNIT_COLUMN,
    CYCLE_COLUMN,
    *OPERATIONAL_SETTING_COLUMNS,
    *SENSOR_COLUMNS,
]

EXPECTED_CMAPSS_COLUMN_COUNT = len(CMAPSS_COLUMNS)

TRAIN_TARGET_COLUMNS = ["rul_uncapped", "rul_capped"]
TEST_TARGET_COLUMN = "test_final_rul"
PREDICTION_COLUMNS = [
    "predicted_rul",
    "residual",
    "absolute_error",
    "squared_error",
]

IDENTIFIER_COLUMNS = [UNIT_COLUMN]

BASE_FEATURE_COLUMNS = [
    *OPERATIONAL_SETTING_COLUMNS,
    *SENSOR_COLUMNS,
]

EXCLUDED_MODEL_INPUT_COLUMNS = [
    UNIT_COLUMN,
    CYCLE_COLUMN,
    *TRAIN_TARGET_COLUMNS,
    TEST_TARGET_COLUMN,
    *PREDICTION_COLUMNS,
]

SENSOR_DESCRIPTIONS = {
    "sensor_1": "T2 total temperature at fan inlet",
    "sensor_2": "T24 total temperature at LPC outlet",
    "sensor_3": "T30 total temperature at HPC outlet",
    "sensor_4": "T50 total temperature at LPT outlet",
    "sensor_5": "P2 pressure at fan inlet",
    "sensor_6": "P15 total pressure in bypass duct",
    "sensor_7": "P30 total pressure at HPC outlet",
    "sensor_8": "Nf physical fan speed",
    "sensor_9": "Nc physical core speed",
    "sensor_10": "Engine pressure ratio",
    "sensor_11": "Ps30 static pressure at HPC outlet",
    "sensor_12": "Fuel flow to Ps30 ratio",
    "sensor_13": "Corrected fan speed",
    "sensor_14": "Corrected core speed",
    "sensor_15": "Bypass ratio",
    "sensor_16": "Burner fuel-air ratio",
    "sensor_17": "Bleed enthalpy",
    "sensor_18": "Demanded fan speed",
    "sensor_19": "Demanded corrected fan speed",
    "sensor_20": "HPT coolant bleed",
    "sensor_21": "LPT coolant bleed",
}

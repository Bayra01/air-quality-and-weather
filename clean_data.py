"""
Улаанбаатарын агаарын чанар + цаг уурын өгөгдлийг цэвэрлэж нэгтгэх скрипт.
Суулгах: pip install pandas numpy
"""
import pandas as pd
import numpy as np

# ===================== ТОХИРГОО (өөрийнхөөрөө солих) =====================
WEATHER_FILE = "weather-data.csv"      # Open-Meteo-оос татсан файл
AIR_FILE = "air-data.csv"            # OpenAQ-аас татсан файл
OUTPUT_FILE = "ub_clean_dataset.csv"

# OpenAQ файлын "parameter" баганаас аль үзүүлэлтийг авах вэ: {файл дахь нэр: шинэ нэр}
AIR_PARAMS = {
    "pm25": "PM2.5",
    "pm10": "PM10",
}
# Зөвхөн нэг станц авах бол нэрийг нь бичнэ (жишээ нь "Bumbugur"). None бол бүх станцыг дундажлана.
LOCATION_NAME = None
# Цаг уурын файлыг Asia/Ulaanbaatar цагийн бүсээр татсан бол энд адилхан үлдээнэ
LOCAL_TZ = "Asia/Ulaanbaatar"
FREQ = "h"                           # цагийн өгөгдөл бол "h", өдрийн бол "D"
MAX_GAP = 6                          # хамгийн ихдээ хэдэн дараалсан хоосныг нөхөх вэ

# Боломжит утгын хүрээ. Үүнээс гадуурх утгыг алдаа гэж үзнэ.
LIMITS = {
    "PM2.5": (0, 1000),
    "PM10": (0, 2000),
    "Temp": (-50, 45),
    "Humidity": (0, 100),
    "Wind": (0, 60),
    "Pressure": (750, 950),          # Улаанбаатарын гадаргуугийн даралт (hPa)
}
# =========================================================================


def load_weather(path):
    # Open-Meteo CSV-ийн эхний 3 мөр нь байршлын мэдээлэл тул алгасна
    df = pd.read_csv(path, skiprows=3)
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "time":
            rename[c] = "time"
        elif lc.startswith("temperature_2m"):
            rename[c] = "Temp"
        elif lc.startswith("relative_humidity"):
            rename[c] = "Humidity"
        elif lc.startswith("wind_speed_10m"):
            rename[c] = "Wind"
        elif lc.startswith("surface_pressure"):
            rename[c] = "Pressure"
        elif lc.startswith("wind_direction_10m"):
            rename[c] = "WindDir"
        elif lc.startswith("precipitation"):
            rename[c] = "Precip"
        elif lc.startswith("snowfall"):
            rename[c] = "Snowfall"
        elif lc.startswith("dew_point"):
            rename[c] = "DewPoint"
    df = df.rename(columns=rename)
    df["time"] = pd.to_datetime(df["time"])
    return df


def load_air(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    print("\nOpenAQ файл дахь станцууд:", df["location_name"].unique().tolist())
    print("Үзүүлэлтүүдийн мөрийн тоо:")
    print(df["parameter"].value_counts())

    if LOCATION_NAME:
        df = df[df["location_name"] == LOCATION_NAME]

    # Зөвхөн хэрэгтэй үзүүлэлтийг авна
    df = df[df["parameter"].isin(AIR_PARAMS.keys())].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # UTC цагийг Улаанбаатарын цаг руу хөрвүүлж, цагийн бүсийн мэдээллийг авна
    t = pd.to_datetime(df["datetimeUtc"], utc=True, errors="coerce")
    df["time"] = t.dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)

    # Урт бүтцийг өргөн бүтэц болгоно: нэг мөр = нэг цаг, багана = үзүүлэлт
    wide = df.pivot_table(index="time", columns="parameter",
                          values="value", aggfunc="mean")
    wide = wide.rename(columns=AIR_PARAMS).reset_index()
    wide.columns.name = None
    return wide


def report(df, title):
    print(f"\n===== {title} =====")
    print("Хэмжээ:", df.shape)
    print("Давхардсан цаг:", df.duplicated(subset="time").sum())
    print("Хоосон утга (тоо / хувь):")
    miss = df.isna().sum()
    print(pd.DataFrame({"тоо": miss, "хувь": (miss / len(df) * 100).round(1)}))


def clean(df, name):
    report(df, f"{name}: ЦЭВЭРЛЭХЭЭС ӨМНӨ")

    # 1. Хугацаа нь уншигдаагүй мөрийг хасна
    df = df.dropna(subset=["time"])

    # 2. Давхардсан цагийг дундажлаж нэгтгэнэ
    df = df.groupby("time", as_index=False).mean(numeric_only=True)

    # 3. Бүх цагийг бүрэн дарааллаар үүсгэж, алга болсон цагийг NaN болгоно
    full = pd.date_range(df["time"].min(), df["time"].max(), freq=FREQ)
    df = df.set_index("time").reindex(full).rename_axis("time")

    # 4. Боломжгүй утгыг (хэт том, сөрөг) хоосон болгоно
    for col, (lo, hi) in LIMITS.items():
        if col in df.columns:
            bad = ((df[col] < lo) | (df[col] > hi)).sum()
            if bad:
                print(f"{name}: {col} дээр хүрээнээс гадуур {bad} утга олдож, хоосон болголоо")
            df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

    # 5. Богино хоосныг (MAX_GAP хүртэл) шугаман нөхнө, урт хоосныг хэвээр үлдээнэ
    df = df.interpolate(method="time", limit=MAX_GAP, limit_area="inside")

    report(df.reset_index(), f"{name}: ЦЭВЭРЛЭСНИЙ ДАРАА")
    return df.reset_index()


def main():
    weather = clean(load_weather(WEATHER_FILE), "Цаг уур")
    air = clean(load_air(AIR_FILE), "Агаарын чанар")

    # Хоёр өгөгдлийг цагаар нь нэгтгэнэ
    data = pd.merge(air, weather, on="time", how="inner")

    # Үндсэн хувьсагч (PM2.5) хоосон мөрийг хасна. Таамаглалын загварт хоосон байж болохгүй.
    before = len(data)
    data = data.dropna(subset=["PM2.5"])
    print(f"\nPM2.5 хоосон байсан {before - len(data)} мөрийг хаслаа")

    # Үлдсэн хоосон утгыг бүрэн нөхөх шаардлагатай бол энд тохируулна
    data = data.dropna()

    # Цагийн шинж чанар нэмнэ (загварт хэрэг болно)
    data["hour"] = data["time"].dt.hour
    data["month"] = data["time"].dt.month
    data["dayofweek"] = data["time"].dt.dayofweek

    data.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nДууслаа: {len(data)} мөр, {OUTPUT_FILE} файлд хадгаллаа")
    print(data.describe().round(2))


if __name__ == "__main__":
    main()
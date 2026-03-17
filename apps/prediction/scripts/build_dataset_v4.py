"""
Build unified beach occupancy dataset from Django export(s) + local cache.

Usage:
    python build_dataset.py --django django_export.json --cache ./cache/predictions --output dataset.csv
    python build_dataset.py --django django_export_old.json django_export.json --cache ./cache/predictions --output dataset.csv
    python build_dataset.py --django django_export_old.json django_export.json --cache ./cache/predictions --beach-profiles beach_profiles.json --output dataset.csv
"""
import argparse
import json
import time
from pathlib import Path
from datetime import timedelta

import pandas as pd
import numpy as np
import holidays

try:
    import weather_module as _weather_module
    HAS_WEATHER = True
except ImportError:
    HAS_WEATHER = False


def load_django_export(path):
    with open(path, "r") as f:
        data = json.load(f)

    rows = []
    for snap in data["snapshots"]:
        rows.append({
            "beach_name": snap["beach_name"],
            "slug": snap.get("camera_slug", snap.get("slug", "")),
            "image_path": snap.get("image_path", ""),
            "prediction_path": snap.get("prediction_path", ""),
            "webcam_id": snap.get("webcam_id", ""),
            "lat": snap["lat"],
            "lon": snap["lon"],
            "ds": pd.to_datetime(snap["ts"]),
            "crowd_count": snap["crowd_count"],
            "source": "django",
        })

    df = pd.DataFrame(rows)

    # Filter out epoch-zero / invalid timestamps
    before = len(df)
    df = df[df["ds"].notna() & (df["ds"] >= "2020-01-01")].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"Django: dropped {dropped} rows with invalid timestamps (epoch zero / null)")

    print(f"Django: {len(df)} snapshots, {df['beach_name'].nunique()} beaches")
    return df


def load_local_cache(cache_dir, model="bayesian_vgg19"):
    model_dir = Path(cache_dir) / model
    if not model_dir.exists():
        print(f"Cache model dir not found: {model_dir}")
        return pd.DataFrame()

    rows = []
    for json_file in model_dir.rglob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            row = {
                "beach_name": data.get("beach", ""),
                "beach_folder": data.get("beach_folder", ""),
                "lat": data.get("lat"),
                "lon": data.get("lon"),
                "ds": pd.to_datetime(data.get("datetime")),
                "crowd_count": data.get("count"),
                "source": "cache",
            }

            weather = data.get("weather", {})
            for k, v in weather.items():
                row[k] = v

            rows.append(row)
        except Exception:
            continue

    df = pd.DataFrame(rows)

    # Filter out epoch-zero / invalid timestamps
    before = len(df)
    df = df[df["ds"].notna() & (df["ds"] >= "2020-01-01")].reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"Cache: dropped {dropped} rows with invalid timestamps (epoch zero / null)")

    print(f"Cache: {len(df)} records, {df['beach_name'].nunique()} beaches")
    return df


def match_beaches(django_df, cache_df, threshold_km=2.0, beach_map_path=None):
    threshold_deg = threshold_km / 111.0

    django_beaches = django_df.groupby("beach_name").agg(
        lat=("lat", "first"), lon=("lon", "first"), slug=("slug", "first"),
        count=("ds", "size"), min_date=("ds", "min"), max_date=("ds", "max")
    ).reset_index()

    cache_beaches = cache_df.groupby("beach_name").agg(
        lat=("lat", "first"), lon=("lon", "first"), beach_folder=("beach_folder", "first"),
        count=("ds", "size"), min_date=("ds", "min"), max_date=("ds", "max")
    ).reset_index()

    # Load manual overrides if provided
    manual_matches = {}
    if beach_map_path and Path(beach_map_path).exists():
        with open(beach_map_path, "r") as f:
            beach_map = json.load(f)
        for entry in beach_map.get("matches", []):
            manual_matches[entry["django_name"]] = entry["cache_name"]
        print(f"Loaded {len(manual_matches)} manual matches from {beach_map_path}")

    matches = []

    # Manual matches first
    for dj_name, ca_name in manual_matches.items():
        dj_row = django_beaches[django_beaches["beach_name"] == dj_name]
        ca_row = cache_beaches[cache_beaches["beach_name"] == ca_name]
        if not dj_row.empty and not ca_row.empty:
            matches.append({
                "django_name": dj_name,
                "cache_name": ca_name,
                "django_slug": dj_row.iloc[0]["slug"],
                "cache_folder": ca_row.iloc[0]["beach_folder"],
                "dist_deg": 0.0,
                "method": "manual",
            })

    already_matched_dj = {m["django_name"] for m in matches}
    already_matched_ca = {m["cache_name"] for m in matches}

    # Auto-match remaining by lat/lon
    for _, dj in django_beaches.iterrows():
        if dj["beach_name"] in already_matched_dj:
            continue
        if dj["lat"] is None or dj["lon"] is None:
            continue
        for _, ca in cache_beaches.iterrows():
            if ca["beach_name"] in already_matched_ca:
                continue
            if ca["lat"] is None or ca["lon"] is None:
                continue
            dist = np.sqrt((dj["lat"] - ca["lat"])**2 + (dj["lon"] - ca["lon"])**2)
            if dist < threshold_deg:
                matches.append({
                    "django_name": dj["beach_name"],
                    "cache_name": ca["beach_name"],
                    "django_slug": dj["slug"],
                    "cache_folder": ca["beach_folder"],
                    "dist_deg": round(dist, 6),
                    "method": "auto",
                })

    matches_df = pd.DataFrame(matches)
    if not matches_df.empty:
        matches_df = matches_df.sort_values("dist_deg").drop_duplicates("django_name", keep="first")
        matches_df = matches_df.sort_values("dist_deg").drop_duplicates("cache_name", keep="first")

    unmatched_django = set(django_beaches["beach_name"]) - set(matches_df["django_name"]) if not matches_df.empty else set(django_beaches["beach_name"])
    unmatched_cache = set(cache_beaches["beach_name"]) - set(matches_df["cache_name"]) if not matches_df.empty else set(cache_beaches["beach_name"])

    # Print matching report
    print(f"\n{'='*60}")
    print("BEACH MATCHING REPORT")
    print(f"{'='*60}")
    print(f"  Django beaches: {len(django_beaches)}")
    print(f"  Cache beaches:  {len(cache_beaches)}")
    print(f"  Matched:        {len(matches_df)}")

    if not matches_df.empty:
        print(f"\n  Matched pairs:")
        for _, m in matches_df.iterrows():
            method_tag = f"[{m['method']}]" if "method" in m else ""
            print(f"    {m['django_name']:30s} ↔ {m['cache_name']:30s} (Δ{m['dist_deg']:.4f}°) {method_tag}")

    if unmatched_django:
        print(f"\n  Django-only beaches (no cache equivalent):")
        for name in sorted(unmatched_django):
            row = django_beaches[django_beaches["beach_name"] == name].iloc[0]
            print(f"    {name:30s} | {row['count']:5d} records | {str(row['min_date'])[:10]} → {str(row['max_date'])[:10]}")

    if unmatched_cache:
        print(f"\n  Cache-only beaches (no Django equivalent):")
        for name in sorted(unmatched_cache):
            row = cache_beaches[cache_beaches["beach_name"] == name].iloc[0]
            print(f"    {name:30s} | {row['count']:5d} records | {str(row['min_date'])[:10]} → {str(row['max_date'])[:10]}")

    return matches_df, unmatched_django, unmatched_cache


def save_beach_map(matches_df, unmatched_django, unmatched_cache, django_df, cache_df, output_path):
    django_beaches = django_df.groupby("beach_name").agg(
        lat=("lat", "first"), lon=("lon", "first"), slug=("slug", "first")
    ).reset_index()
    cache_beaches = cache_df.groupby("beach_name").agg(
        lat=("lat", "first"), lon=("lon", "first"), beach_folder=("beach_folder", "first")
    ).reset_index()

    beach_map = {
        "matches": [],
        "django_only": [],
        "cache_only": [],
    }

    if not matches_df.empty:
        for _, m in matches_df.iterrows():
            beach_map["matches"].append({
                "django_name": m["django_name"],
                "cache_name": m["cache_name"],
                "django_slug": m.get("django_slug", ""),
                "cache_folder": m.get("cache_folder", ""),
                "dist_deg": m["dist_deg"],
                "method": m.get("method", "auto"),
            })

    for name in sorted(unmatched_django):
        row = django_beaches[django_beaches["beach_name"] == name]
        if not row.empty:
            r = row.iloc[0]
            beach_map["django_only"].append({
                "beach_name": name, "slug": r.get("slug", ""),
                "image_path": r.get("image_path", ""),
                "prediction_path": r.get("prediction_path", ""),
                "lat": float(r["lat"]) if r["lat"] else None,
                "lon": float(r["lon"]) if r["lon"] else None,
            })

    for name in sorted(unmatched_cache):
        row = cache_beaches[cache_beaches["beach_name"] == name]
        if not row.empty:
            r = row.iloc[0]
            beach_map["cache_only"].append({
                "beach_name": name, "beach_folder": r.get("beach_folder", ""),
                "lat": float(r["lat"]) if r["lat"] else None,
                "lon": float(r["lon"]) if r["lon"] else None,
            })

    with open(output_path, "w") as f:
        json.dump(beach_map, f, indent=2, ensure_ascii=False)
    print(f"\nBeach map saved → {output_path}")
    print("  Edit 'matches' to add manual overrides, then re-run with --beach-map")


def get_weather_columns(cache_df):
    return [c for c in cache_df.columns if c.startswith(("ae_", "om_"))]


def merge_datasets(django_df, cache_df, matches_df):
    weather_cols = get_weather_columns(cache_df) if not cache_df.empty else []
    IMG_COLS = ["image_path", "prediction_path"]

    def _add_img_cols(df):
        for c in IMG_COLS:
            if c not in df.columns:
                df[c] = ""
        return df

    # Handle single-source cases
    if django_df.empty and not cache_df.empty:
        result = cache_df.copy()
        result["unique_id"] = result["beach_name"]
        if result["ds"].dt.tz is not None:
            result["ds"] = result["ds"].dt.tz_convert("UTC").dt.tz_localize(None)
        result["ds"] = result["ds"].dt.floor("h")
        result["y"] = result["crowd_count"]
        result["source"] = "cache"
        id_cols = [c for c in ["beach_folder"] if c in result.columns]
        for c in ["slug", "webcam_id"]:
            result[c] = ""
        if "beach_folder" not in result.columns:
            result["beach_folder"] = ""
        result = _add_img_cols(result)
        result = result[["unique_id", "beach_name", "ds", "y", "lat", "lon", "source", "slug", "webcam_id", "beach_folder"] + IMG_COLS + weather_cols]
        return result.drop_duplicates(subset=["unique_id", "ds"], keep="first")

    if cache_df.empty and not django_df.empty:
        result = django_df.copy()
        result["unique_id"] = result["slug"]
        if result["ds"].dt.tz is not None:
            result["ds"] = result["ds"].dt.tz_convert("UTC").dt.tz_localize(None)
        result["ds"] = result["ds"].dt.floor("h")
        result["y"] = result["crowd_count"]
        result["source"] = "django"
        for c in ["slug", "webcam_id"]:
            if c not in result.columns:
                result[c] = ""
        result["beach_folder"] = ""
        result = _add_img_cols(result)
        result = result[["unique_id", "beach_name", "ds", "y", "lat", "lon", "source", "slug", "webcam_id", "beach_folder"] + IMG_COLS]
        return result.drop_duplicates(subset=["unique_id", "ds"], keep="first")

    # Normalize beach names using matches
    name_map_django = {}
    name_map_cache = {}
    if not matches_df.empty:
        for _, m in matches_df.iterrows():
            canonical = m["django_name"]
            name_map_django[m["django_name"]] = canonical
            name_map_cache[m["cache_name"]] = canonical

    # Apply canonical names
    django_df = django_df.copy()
    cache_df = cache_df.copy()
    django_df["unique_id"] = django_df["beach_name"].map(name_map_django).fillna(django_df["beach_name"])
    cache_df["unique_id"] = cache_df["beach_name"].map(name_map_cache).fillna(cache_df["beach_name"])

    # Normalize timezones to naive UTC
    for d in [django_df, cache_df]:
        if d["ds"].dt.tz is not None:
            d["ds"] = d["ds"].dt.tz_convert("UTC").dt.tz_localize(None)
    django_df["ds_hour"] = django_df["ds"].dt.floor("h")
    cache_df["ds_hour"] = cache_df["ds"].dt.floor("h")

    # For matched beaches: merge Django counts with cache weather
    matched_names = set(name_map_django.values())

    dj_matched = django_df[django_df["unique_id"].isin(matched_names)].copy()
    ca_matched = cache_df[cache_df["unique_id"].isin(matched_names)].copy()

    # Merge on (unique_id, ds_hour) — outer join to keep all records
    if not dj_matched.empty and not ca_matched.empty:
        dj_cols = ["unique_id", "beach_name", "ds_hour", "crowd_count", "lat", "lon"]
        if "slug" in dj_matched.columns:
            dj_cols.append("slug")
        if "webcam_id" in dj_matched.columns:
            dj_cols.append("webcam_id")
        if 'image_path' in dj_matched.columns:
            dj_cols.append('image_path')
        if 'prediction_path'  in dj_matched.columns:
            dj_cols.append('prediction_path')


        ca_cols = ["unique_id", "ds_hour", "crowd_count", "lat", "lon"] + weather_cols
        if "beach_folder" in ca_matched.columns:
            ca_cols.append("beach_folder")

        merged = pd.merge(
            dj_matched[dj_cols].rename(
                columns={"crowd_count": "count_django", "lat": "lat_dj", "lon": "lon_dj"}
            ),
            ca_matched[ca_cols].rename(
                columns={"crowd_count": "count_cache", "lat": "lat_ca", "lon": "lon_ca"}
            ),
            on=["unique_id", "ds_hour"],
            how="outer",
        )
        merged["y"] = merged["count_django"].fillna(merged["count_cache"])
        merged["lat"] = merged["lat_dj"].fillna(merged["lat_ca"])
        merged["lon"] = merged["lon_dj"].fillna(merged["lon_ca"])
        merged["source"] = np.where(
            merged["count_django"].notna() & merged["count_cache"].notna(), "both",
            np.where(merged["count_django"].notna(), "django", "cache")
        )
        for c in ["slug", "webcam_id", "beach_folder", "beach_name"] + IMG_COLS:
            if c not in merged.columns:
                merged[c] = ""
            else:
                merged[c] = merged[c].fillna("")
        merged = merged.drop(columns=["count_django", "count_cache", "lat_dj", "lon_dj", "lat_ca", "lon_ca"])
    else:
        merged = pd.DataFrame()

    # Unmatched Django beaches (no weather)
    dj_unmatched = django_df[~django_df["unique_id"].isin(matched_names)].copy()
    if not dj_unmatched.empty:
        dj_unmatched = dj_unmatched.rename(columns={"ds_hour": "ds_hour"})
        dj_unmatched["y"] = dj_unmatched["crowd_count"]
        dj_unmatched["source"] = "django"
        for col in weather_cols:
            dj_unmatched[col] = np.nan
        for c in ["slug", "webcam_id"]:
            if c not in dj_unmatched.columns:
                dj_unmatched[c] = ""
        dj_unmatched["beach_folder"] = ""
        dj_unmatched = _add_img_cols(dj_unmatched)
        dj_unmatched = dj_unmatched[["unique_id", "beach_name", "ds_hour", "y", "lat", "lon", "source", "slug", "webcam_id", "beach_folder"] + IMG_COLS + weather_cols]

    # Unmatched cache beaches (have weather)
    ca_unmatched = cache_df[~cache_df["unique_id"].isin(matched_names)].copy()
    if not ca_unmatched.empty:
        ca_unmatched["y"] = ca_unmatched["crowd_count"]
        ca_unmatched["source"] = "cache"
        ca_unmatched["slug"] = ""
        ca_unmatched["webcam_id"] = ""
        if "beach_folder" not in ca_unmatched.columns:
            ca_unmatched["beach_folder"] = ""
        ca_unmatched = _add_img_cols(ca_unmatched)
        ca_unmatched = ca_unmatched[["unique_id", "beach_name", "ds_hour", "y", "lat", "lon", "source", "slug", "webcam_id", "beach_folder"] + IMG_COLS + weather_cols]

    # Concatenate all
    parts = [df for df in [merged, dj_unmatched, ca_unmatched] if not df.empty]
    if not parts:
        print("No data to merge!")
        return pd.DataFrame()

    result = pd.concat(parts, ignore_index=True)
    result = result.rename(columns={"ds_hour": "ds"})
    result = result.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    # Clean identifier columns
    for c in ["slug", "webcam_id", "beach_folder", "beach_name"] + ["image_path", "prediction_path"]:
        if c in result.columns:
            result[c] = result[c].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)

    # Use slug (camera_slug) as unique_id when available, keeps per-camera series
    has_slug = result["slug"].replace("", np.nan).notna()
    result.loc[has_slug, "unique_id"] = result.loc[has_slug, "slug"]

    # Fill empty beach_name from unique_id (cache records have no separate name)
    empty_name = result["beach_name"].replace("", np.nan).isna()
    result.loc[empty_name, "beach_name"] = result.loc[empty_name, "unique_id"]

    # Remove exact duplicates (same camera + same hour)
    result = result.drop_duplicates(subset=["unique_id", "ds"], keep="first")

    return result


def add_temporal_features(df):
    df = df.copy()
    df["hour"] = df["ds"].dt.hour
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    df["day_of_year"] = df["ds"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Holidays: Spanish national + Balearic Islands regional
    years = sorted(df["ds"].dt.year.unique())
    es_holidays = holidays.Spain(prov="IB", years=years)

    # Bridge days: weekday squeezed between holiday and weekend
    bridge_days = {}
    for d, name in es_holidays.items():
        dow = d.weekday()
        if dow == 3:  # Thu holiday → Fri bridge
            bridge_days[d + timedelta(days=1)] = f"Puente: {name}"
        elif dow == 1:  # Tue holiday → Mon bridge
            bridge_days[d - timedelta(days=1)] = f"Puente: {name}"

    all_special = {pd.Timestamp(k): v for k, v in es_holidays.items()}
    all_special.update({pd.Timestamp(k): v for k, v in bridge_days.items()})

    dates = df["ds"].dt.normalize()
    df["holiday_name"] = dates.map(all_special).fillna("")
    df["is_holiday"] = (df["holiday_name"] != "").astype(int)

    n_h = df["is_holiday"].sum()
    unique_h = df.loc[df["is_holiday"] == 1, "holiday_name"].nunique()
    print(f"  Holidays: {n_h:,} records flagged ({unique_h} unique holidays/bridges)")

    return df


BEACH_ENCODINGS = {
    "grado_de_ocupacion": {"LOW": 0, "MEDIUM": 1, "HIGH": 2},
    "proximidad_al_nucleo_urbano": {"REMOTE": 0, "SEMI_URBAN": 1, "URBAN": 2},
    "composicion_de_la_playa": {"SAND": 0, "GRAVEL": 1, "PEBBLES": 2, "ROCKS": 3},
    "condiciones_de_bano": {"CALM": 0, "MODERATE": 1, "STRONG": 2},
}

BOOL_FIELDS = ["paseo_maritimo", "tipo_de_usuario_local", "tipo_de_usuario_turista"]


def load_beach_profiles(path):
    with open(path, "r") as f:
        profiles = json.load(f)
    print(f"Beach profiles: {len(profiles)} cameras from {path}")
    return profiles


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def add_beach_metadata(df, profiles, threshold_km=2.0):
    if not profiles:
        return df

    df = df.copy()
    profile_list = [(slug, p) for slug, p in profiles.items() if p.get("lat") and p.get("lon")]

    beach_coords = df.groupby("unique_id")[["lat", "lon"]].first()
    matched = 0

    for uid, row in beach_coords.iterrows():
        best_dist, best_profile = threshold_km, None
        for slug, p in profile_list:
            d = _haversine_km(row["lat"], row["lon"], p["lat"], p["lon"])
            if d < best_dist:
                best_dist, best_profile = d, p

        if best_profile is None:
            continue

        matched += 1
        mask = df["unique_id"] == uid

        for field, encoding in BEACH_ENCODINGS.items():
            val = best_profile.get(field)
            df.loc[mask, f"stat_{field}"] = encoding.get(val, np.nan) if val else np.nan

        for field in BOOL_FIELDS:
            val = best_profile.get(field)
            df.loc[mask, f"stat_{field}"] = int(val) if val is not None else np.nan

    print(f"\nBeach metadata: {matched}/{len(beach_coords)} beaches matched (threshold={threshold_km}km)")
    stat_cols = [c for c in df.columns if c.startswith("stat_") and c not in ["stat_mean_y", "stat_cv"]]
    for col in stat_cols:
        pct = df[col].notna().mean() * 100
        print(f"  {col:45s} {pct:.0f}%")

    return df


def _normalize_django_slugs(df, threshold_km=0.5):
    """Unify slugs across exports when same beach has different slug names."""
    slug_coords = df.groupby("slug")[["lat", "lon"]].first()
    slugs = list(slug_coords.index)

    canonical = {}
    for s in slugs:
        canonical[s] = s

    for i, s1 in enumerate(slugs):
        for s2 in slugs[i + 1:]:
            if canonical[s2] != s2:
                continue
            d = _haversine_km(
                slug_coords.loc[s1, "lat"], slug_coords.loc[s1, "lon"],
                slug_coords.loc[s2, "lat"], slug_coords.loc[s2, "lon"],
            )
            if d < threshold_km:
                canonical[s2] = canonical[s1]

    renamed = {s: c for s, c in canonical.items() if s != c}
    if renamed:
        print(f"\nSlug normalization ({len(renamed)} renamed):")
        for old_s, new_s in renamed.items():
            old_name = df[df["slug"] == old_s]["beach_name"].iloc[0]
            new_name = df[df["slug"] == new_s]["beach_name"].iloc[0]
            print(f"  {old_s:45s} → {new_s:45s}  ({old_name} → {new_name})")
        df["slug"] = df["slug"].map(canonical)

    return df


def print_summary(df):
    print(f"\n{'='*60}")
    print(f"UNIFIED DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total records:   {len(df)}")
    print(f"Date range:      {df['ds'].min()} → {df['ds'].max()}")
    print(f"Beaches:         {df['unique_id'].nunique()}")
    print(f"Source breakdown: {df['source'].value_counts().to_dict()}")

    weather_cols = [c for c in df.columns if c.startswith(("ae_", "om_"))]
    weather_coverage = df[weather_cols].notna().any(axis=1).sum()
    print(f"Weather coverage: {weather_coverage}/{len(df)} ({100*weather_coverage/len(df):.1f}%)")

    if "image_path" in df.columns:
        img_count = (df["image_path"].replace("", np.nan).notna()).sum()
        print(f"Image paths:     {img_count}/{len(df)} ({100*img_count/len(df):.1f}%)")
    if "prediction_path" in df.columns:
        pred_count = (df["prediction_path"].replace("", np.nan).notna()).sum()
        print(f"Prediction paths:{pred_count}/{len(df)} ({100*pred_count/len(df):.1f}%)")

    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    if flag_cols:
        clean = (df[flag_cols].sum(axis=1) == 0).sum()
        print(f"Clean records:   {clean}/{len(df)} ({100*clean/len(df):.1f}%)")

    print(f"\nPer beach:")
    for name, group in df.groupby("unique_id"):
        weather_pct = group[weather_cols].notna().any(axis=1).mean() * 100 if weather_cols else 0
        sources = group["source"].value_counts().to_dict()
        source_str = " | ".join(f"{k}:{v}" for k, v in sorted(sources.items()))
        flags = ""
        if flag_cols:
            active_flags = [c.replace("flag_", "") for c in flag_cols if group[c].sum() > 0]
            if active_flags:
                flags = f" ⚠ {','.join(active_flags)}"

        # Identifiers
        ids = []
        if "webcam_id" in group.columns:
            wid = group["webcam_id"].replace("", np.nan).dropna().unique()
            if len(wid) > 0:
                ids.append(f"webcam={','.join(str(w) for w in wid)}")
        if "slug" in group.columns:
            slugs = group["slug"].replace("", np.nan).dropna().unique()
            if len(slugs) > 0:
                ids.append(f"slug={slugs[0]}")
        if "beach_folder" in group.columns:
            folders = group["beach_folder"].replace("", np.nan).dropna().unique()
            if len(folders) > 0:
                ids.append(f"folder={','.join(folders)}")
        id_str = f" [{' | '.join(ids)}]" if ids else ""

        print(f"  {name:30s} | {len(group):6d} records | "
              f"{group['ds'].min().date()} → {group['ds'].max().date()} | "
              f"weather: {weather_pct:.0f}% | {source_str}{flags}{id_str}")


def clean_weather_cache(cache_dir="cache/weather"):
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"Weather cache not found: {cache_path}")
        return 0

    removed = 0
    for json_file in cache_path.glob("om_*.json"):
        try:
            with open(json_file, "r") as f:
                json.load(f)
        except (json.JSONDecodeError, ValueError):
            json_file.unlink()
            removed += 1

    print(f"Cleaned {removed} corrupted Open-Meteo cache files from {cache_path}")
    return removed


OPENMETEO_VARS = [
    'temperature_2m', 'apparent_temperature', 'dewpoint_2m', 'relative_humidity_2m',
    'pressure_msl', 'precipitation', 'rain', 'wind_speed_10m', 'wind_direction_10m',
    'wind_gusts_10m', 'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high',
    'sunshine_duration', 'vapour_pressure_deficit', 'direct_radiation', 'shortwave_radiation'
]


def _bulk_fetch_openmeteo(lat, lon, start_date, end_date, delay=1.5):
    import requests

    # Open-Meteo archive API allows large date ranges in one call
    # Split into yearly chunks to avoid response size issues
    all_data = {}
    current_start = start_date

    while current_start <= end_date:
        current_end = min(end_date, current_start.replace(year=current_start.year + 1) - pd.Timedelta(days=1))

        try:
            time.sleep(delay)
            resp = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={
                    'latitude': round(lat, 4),
                    'longitude': round(lon, 4),
                    'start_date': str(current_start),
                    'end_date': str(current_end),
                    'hourly': ','.join(OPENMETEO_VARS),
                    'timezone': 'UTC'
                },
                timeout=60
            )
            resp.raise_for_status()
            data = resp.json()

            if 'hourly' in data:
                times = pd.to_datetime(data['hourly']['time'])
                for i, t in enumerate(times):
                    row = {}
                    for var in OPENMETEO_VARS:
                        if var in data['hourly']:
                            val = data['hourly'][var][i]
                            if val is not None:
                                row[f'om_{var}'] = val
                    if row:
                        all_data[t] = row

            print(f"    Fetched {current_start} → {current_end}: {len(data.get('hourly', {}).get('time', []))} hours")

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"    Rate limited, waiting 30s...")
                time.sleep(30)
                continue  # retry same chunk
            print(f"    HTTP error {current_start}→{current_end}: {e}")
        except Exception as e:
            print(f"    Error {current_start}→{current_end}: {e}")

        current_start = current_end + pd.Timedelta(days=1)

    return all_data


def _bulk_aemet_interpolate(locations_dates, aemet_module):
    """Batch AEMET interpolation for multiple (lat, lon, datetime) tuples."""
    results = {}

    try:
        df_aemet = aemet_module._load_aemet_data()
        if df_aemet is None or len(df_aemet) == 0:
            return results
    except Exception:
        return results

    # Group by date to reuse time-filtered data
    by_date = {}
    for lat, lon, dt in locations_dates:
        d = dt.date() if hasattr(dt, 'date') else dt
        by_date.setdefault(d, []).append((lat, lon, dt))

    for date_key, entries in by_date.items():
        time_data = df_aemet[df_aemet['fint'].dt.date == date_key]
        if len(time_data) == 0:
            continue

        available_vars = [v for v in aemet_module.AEMET_VARS if v in time_data.columns]
        if not available_vars:
            continue

        stations = time_data.groupby(['idema', 'lat', 'lon']).agg(
            {v: 'mean' for v in available_vars}
        ).reset_index()

        for lat, lon, dt in entries:
            target = np.array([[lon, lat]])
            row = {}
            for var in available_vars:
                var_stations = stations[['lon', 'lat', var]].dropna()
                if len(var_stations) < 3:
                    if len(var_stations) > 0:
                        row[f'ae_{var}'] = round(float(var_stations[var].mean()), 4)
                    continue
                try:
                    val = aemet_module._hull_multipoint_interpolate(
                        var_stations[['lon', 'lat']].values,
                        var_stations[var].values,
                        target
                    )
                    if val is not None:
                        row[f'ae_{var}'] = round(val, 4)
                except Exception:
                    pass
            if row:
                results[(lat, lon, dt)] = row

    return results


def enrich_weather(df, delay=1.5, save_every=500, output_path=None):
    weather_cols = [c for c in df.columns if c.startswith(("ae_", "om_"))]
    has_weather = df[weather_cols].notna().any(axis=1) if weather_cols else pd.Series(False, index=df.index)
    missing_idx = df.index[~has_weather].tolist()

    if not missing_idx:
        print("All records already have weather data.")
        return df

    missing = df.loc[missing_idx]

    # Find unique locations
    locations = missing.groupby(
        [missing["lat"].round(4), missing["lon"].round(4)]
    ).agg(
        min_date=("ds", "min"),
        max_date=("ds", "max"),
        count=("ds", "size")
    ).reset_index()

    print(f"\nBulk weather enrichment:")
    print(f"  {len(missing_idx)} records missing weather")
    print(f"  {len(locations)} unique locations to fetch")

    # Phase 1: Bulk fetch Open-Meteo (one call per location per year)
    print(f"\n  Phase 1: Open-Meteo bulk fetch...")
    om_lookup = {}  # (lat_round, lon_round) -> {datetime -> {om_vars}}

    for _, loc in locations.iterrows():
        lat, lon = loc["lat"], loc["lon"]
        start = loc["min_date"].date()
        end = loc["max_date"].date()
        print(f"  Location ({lat:.4f}, {lon:.4f}): {start} → {end} ({loc['count']} records)")

        data = _bulk_fetch_openmeteo(lat, lon, start, end, delay=delay)
        om_lookup[(round(lat, 4), round(lon, 4))] = data

    # Phase 2: AEMET batch interpolation
    aemet_results = {}
    if HAS_WEATHER:
        print(f"\n  Phase 2: AEMET batch interpolation...")
        try:
            locations_dates = [
                (row["lat"], row["lon"], row["ds"].to_pydatetime())
                for _, row in missing.iterrows()
            ]
            aemet_results = _bulk_aemet_interpolate(locations_dates, _weather_module)
            print(f"    AEMET: {len(aemet_results)} results")
        except Exception as e:
            print(f"    AEMET skipped: {e}")
    else:
        print(f"\n  Phase 2: AEMET skipped (weather_module not available)")

    # Phase 3: Apply to DataFrame
    print(f"\n  Phase 3: Applying to dataset...")
    filled = 0
    for idx in missing_idx:
        row = df.loc[idx]
        lat_r = round(row["lat"], 4)
        lon_r = round(row["lon"], 4)
        ds_hour = row["ds"].floor("h")

        weather = {}

        # Open-Meteo lookup
        loc_data = om_lookup.get((lat_r, lon_r), {})
        if ds_hour in loc_data:
            weather.update(loc_data[ds_hour])

        # AEMET lookup
        aemet_key = (row["lat"], row["lon"], row["ds"].to_pydatetime())
        if aemet_key in aemet_results:
            weather.update(aemet_results[aemet_key])

        if weather:
            for k, v in weather.items():
                if k not in df.columns:
                    df[k] = np.nan
                df.at[idx, k] = v
            filled += 1

    print(f"  Done: {filled}/{len(missing_idx)} records enriched")
    return df


def main():
    parser = argparse.ArgumentParser(description="Build unified beach dataset")
    parser.add_argument("--django", type=str, nargs="+", required=True, help="Path(s) to django_export JSON file(s)")
    parser.add_argument("--cache", type=str, required=False, help="Path to cache/predictions directory")
    parser.add_argument("--model", type=str, default="bayesian_vgg19")
    parser.add_argument("--output", type=str, default="unified_dataset.csv")
    parser.add_argument("--threshold-km", type=float, default=2.0, help="Beach matching distance threshold in km")
    parser.add_argument("--enrich-weather", action="store_true", help="Fetch weather for records missing it", default=True)
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds between Open-Meteo API calls")
    parser.add_argument("--clean-weather-cache", action="store_true", help="Remove corrupted Open-Meteo cache files before enriching")
    parser.add_argument("--save-beach-map", type=str, default=None, help="Save beach matching to JSON for review")
    parser.add_argument("--beach-map", type=str, default=None, help="Load manual beach matching overrides from JSON")
    parser.add_argument("--beach-profiles", type=str, default=None, help="Beach metadata JSON from export_beach_profiles.py")
    args = parser.parse_args()

    dfs = [load_django_export(p) for p in args.django]
    non_empty = [d for d in dfs if not d.empty]
    django_df = pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()
    if not django_df.empty:
        if len(non_empty) > 1:
            django_df = _normalize_django_slugs(django_df, threshold_km=0.5)
        before = len(django_df)
        django_df = django_df.drop_duplicates(subset=["slug", "ds"], keep="first").reset_index(drop=True)
        dupes = before - len(django_df)
        if dupes:
            print(f"Django: dropped {dupes} duplicates across files")
        print(f"Django total: {len(django_df)} snapshots, {django_df['beach_name'].nunique()} beaches")

    cache_df = pd.DataFrame()
    if args.cache:
        cache_df = load_local_cache(args.cache, model=args.model)

    if django_df.empty and cache_df.empty:
        print("No data found!")
        return

    matches_df = pd.DataFrame()
    unmatched_dj = set()
    unmatched_ca = set()
    if not django_df.empty and not cache_df.empty:
        matches_df, unmatched_dj, unmatched_ca = match_beaches(
            django_df, cache_df, threshold_km=args.threshold_km, beach_map_path=args.beach_map
        )
    elif not django_df.empty:
        unmatched_dj = set(django_df["beach_name"].unique())
        print(f"\nNo cache data — all {len(unmatched_dj)} Django beaches are standalone")
    elif not cache_df.empty:
        unmatched_ca = set(cache_df["beach_name"].unique())
        print(f"\nNo Django data — all {len(unmatched_ca)} cache beaches are standalone")

    if args.save_beach_map:
        save_beach_map(matches_df, unmatched_dj, unmatched_ca, django_df, cache_df, args.save_beach_map)

    df = merge_datasets(django_df, cache_df, matches_df)
    if df.empty:
        print("Merge produced empty dataset!")
        return

    if args.enrich_weather:
        if args.clean_weather_cache:
            clean_weather_cache()
        df = enrich_weather(df, delay=args.delay)

    df = add_temporal_features(df)

    if args.beach_profiles:
        profiles = load_beach_profiles(args.beach_profiles)
        df = add_beach_metadata(df, profiles, threshold_km=args.threshold_km)

    print_summary(df)

    df.to_csv(args.output, index=False)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
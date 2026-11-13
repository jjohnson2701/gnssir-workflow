# ABOUTME: Downloads daily RINEX 3 files of one station from EarthScope and translates them to SNR files.
# ABOUTME: For stations whose multi-GNSS data are in the EarthScope rinex3 archive (15 s files).
"""EarthScope RINEX 3 to SNR files.

For each day: download the compact RINEX 3 file, unpack it with CRX2RNX, and run
`rinex2snr -input_file`. The unpacked file is removed afterwards.

You need a free EarthScope account and a saved login. Make it once with:
    es login          (from the earthscope-cli package)
The token is read from the saved login and is never printed.

Usage:
    python -m scripts.earthscope_rinex3 --station sc02 --year 2024 --doy 60 --doy_end 91 --keep downloads/
    (REFL_CODE, ORBITS and EXE must be set; CRX2RNX must be in $EXE or on PATH)
"""
import argparse
import gzip
import os
import shutil
import subprocess
import time
from pathlib import Path

URL = "https://gage-data.earthscope.org/archive/gnss/rinex3/obs/{year}/{doy:03d}/{name}"


def file_name(station9, year, doy, interval="15S"):
    return f"{station9}_R_{year}{doy:03d}0000_01D_{interval}_MO.crx.gz"


def access_token():
    from earthscope_sdk.auth.device_code_flow import DeviceCodeFlowSimple

    return DeviceCodeFlowSimple(Path.home() / ".config/earthscope").get_access_token_refresh_if_necessary()


def crx2rnx_path():
    for cand in (Path(os.environ.get("EXE", "")) / "CRX2RNX", shutil.which("CRX2RNX")):
        if cand and Path(cand).exists():
            return str(cand)
    raise SystemExit("CRX2RNX not found in $EXE or on PATH.")


def download(station9, year, doy, out_dir, interval="15S", tries=3):
    """Download one day. Returns the path, or None when the archive has no file."""
    import requests

    name = file_name(station9, year, doy, interval)
    path = Path(out_dir) / name
    if path.exists() and path.stat().st_size > 100_000:
        return path
    for _ in range(tries):
        try:
            r = requests.get(URL.format(year=year, doy=doy, name=name),
                             headers={"Authorization": f"Bearer {access_token()}"}, timeout=300)
        except requests.RequestException:
            time.sleep(5)
            continue
        if r.status_code == 200:
            path.write_bytes(r.content)
            return path
        if r.status_code in (403, 404):
            return None
        time.sleep(5)
    return None


def translate(crx_gz, work_dir, snr_type=66):
    """Unpack one compact RINEX 3 file and run rinex2snr on it. Returns True on success."""
    work_dir = Path(work_dir)
    crx = work_dir / Path(crx_gz).name[:-3]
    with gzip.open(crx_gz, "rb") as src, open(crx, "wb") as dst:
        shutil.copyfileobj(src, dst)
    rnx = crx.with_suffix(".rnx")
    try:
        subprocess.run([crx2rnx_path(), "-f", str(crx)], check=True, cwd=work_dir)
        done = subprocess.run(["rinex2snr", "-input_file", str(rnx), "-snr", str(snr_type)], cwd=work_dir,
                              capture_output=True, text=True)
        return "SUCCESS" in done.stdout
    finally:
        for f in (crx, rnx):
            f.unlink(missing_ok=True)


def main(argv=None):
    p = argparse.ArgumentParser(description="Download EarthScope RINEX 3 files and make SNR files.")
    p.add_argument("--station", required=True, help="4-character name, for example sc02")
    p.add_argument("--long_name", help="9-character RINEX 3 name; default <STATION>00USA")
    p.add_argument("--year", type=int, required=True)
    p.add_argument("--doy", type=int, required=True)
    p.add_argument("--doy_end", type=int)
    p.add_argument("--interval", default="15S", help="sample interval in the file name (default 15S)")
    p.add_argument("--keep", default="earthscope_rinex3", help="folder for the downloaded files")
    a = p.parse_args(argv)
    for var in ("REFL_CODE", "ORBITS"):
        if not os.environ.get(var):
            raise SystemExit(f"Set {var}.")
    station9 = (a.long_name or f"{a.station.upper()}00USA").upper()
    keep = Path(a.keep)
    keep.mkdir(parents=True, exist_ok=True)
    ok, missing, failed = 0, [], []
    for doy in range(a.doy, (a.doy_end or a.doy) + 1):
        f = download(station9, a.year, doy, keep, a.interval)
        if f is None:
            missing.append(doy)
            continue
        if translate(f, keep):
            ok += 1
        else:
            failed.append(doy)
    print(f"{ok} days translated to SNR files in {os.environ['REFL_CODE']}/{a.year}/snr/{a.station.lower()}/")
    if missing:
        print("No file in the archive for days:", ", ".join(f"{d:03d}" for d in missing))
    if failed:
        print("Translation failed for days:", ", ".join(f"{d:03d}" for d in failed))


if __name__ == "__main__":
    main()

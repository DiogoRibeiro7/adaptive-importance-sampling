#!/usr/bin/env python
"""Fetch a USGS annual peak-streamflow record and write it as a tidy CSV.

The National Water Information System publishes, for each gauge, the largest
instantaneous discharge observed in each water year. That is the quantity flood
frequency analysis is built on, and it is what
``data/usgs_01646500_annual_peaks.csv`` holds.

Run it to refresh the file or to fetch a different gauge::

    python scripts/fetch_usgs_peaks.py --site 01646500
    python scripts/fetch_usgs_peaks.py --site 01646500 --output data/potomac.csv

The service needs no key. USGS data are in the public domain as works of the
United States government.
"""

from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from pathlib import Path

PEAK_SERVICE = "https://nwis.waterdata.usgs.gov/nwis/peak"

#: Cubic feet per second to cubic metres per second.
CFS_TO_CUMECS = 0.028316846592


def fetch(site: str, timeout: float = 60.0) -> str:
    """Retrieve the RDB-formatted peak record for one gauge."""
    url = f"{PEAK_SERVICE}?site_no={site}&agency_cd=USGS&format=rdb"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return str(response.read().decode("utf-8", "replace"))


def water_year(date: str) -> int:
    """Water year containing ``date``, given as ``YYYY-MM-DD``.

    A USGS water year runs from 1 October to 30 September and is named for the
    calendar year it ends in, so a peak in October 2011 belongs to water year
    2012. Taking the calendar year instead produces a record with some years
    twice and others missing: on this gauge it gave 95 rows spanning only 80
    distinct labels, with 1934, 1937 and 2011 each appearing twice.
    """
    year, month = int(date[:4]), int(date[5:7])
    return year + 1 if month >= 10 else year


def parse(text: str) -> list[dict[str, str]]:
    """Pull the year, discharge and gauge height out of the RDB payload.

    RDB is tab separated with ``#`` comments, a header row, and a row of format
    codes beneath it. Rows without a discharge are dropped: the record marks
    some years as unmeasured, and a blank there is missing data rather than a
    zero flood.
    """
    rows: list[dict[str, str]] = []
    header: list[str] | None = None

    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if header is None:
            header = fields
            continue
        if fields and fields[0].endswith("s") and fields[0][:-1].isdigit():
            continue  # the format-code row, e.g. "5s", "15s"

        record = dict(zip(header, fields, strict=False))
        date = record.get("peak_dt", "")
        discharge = record.get("peak_va", "").strip()
        if not date or not discharge:
            continue

        rows.append(
            {
                "water_year": str(water_year(date)),
                "peak_date": date,
                "peak_discharge_cfs": discharge,
                "gage_height_ft": record.get("gage_ht", "").strip(),
            }
        )

    return rows


def write(rows: list[dict[str, str]], destination: Path, site: str) -> None:
    """Write the tidy CSV, adding discharge in cubic metres per second."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "water_year",
                "peak_date",
                "peak_discharge_cfs",
                "peak_discharge_m3s",
                "gage_height_ft",
            ]
        )
        for row in rows:
            cfs = float(row["peak_discharge_cfs"])
            writer.writerow(
                [
                    row["water_year"],
                    row["peak_date"],
                    row["peak_discharge_cfs"],
                    f"{cfs * CFS_TO_CUMECS:.3f}",
                    row["gage_height_ft"],
                ]
            )
    print(f"Wrote {len(rows)} annual peaks for gauge {site} to {destination}")


def main(argv: list[str] | None = None) -> int:
    """Fetch one gauge's annual peaks and write them to CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--site",
        default="01646500",
        help="USGS site number (default: 01646500, Potomac River at Little Falls)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination CSV (default: data/usgs_<site>_annual_peaks.csv)",
    )
    args = parser.parse_args(argv)

    destination = args.output or Path("data") / f"usgs_{args.site}_annual_peaks.csv"

    try:
        rows = parse(fetch(args.site))
    except OSError as error:
        print(f"Could not reach the USGS service: {error}", file=sys.stderr)
        return 1

    if not rows:
        print(f"No peak records found for site {args.site}.", file=sys.stderr)
        return 1

    write(rows, destination, args.site)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

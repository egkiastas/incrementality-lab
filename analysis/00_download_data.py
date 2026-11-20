"""Download the Criteo file (checksum-verified) and convert it to Parquet.

With --fixture, write the small synthetic dataset used by the tests instead.
"""

from incrementality import data
from incrementality.config import Config, from_args
from incrementality.synthetic import write_fixture


def main(cfg: Config, fixture: bool = False) -> None:
    if fixture:
        write_fixture(cfg.parquet, n=cfg.expected_rows)
    elif not cfg.parquet.exists():
        data.convert_to_parquet(data.download(cfg.raw_csv), cfg.parquet, cfg.expected_rows)
    print(f"data ready: {cfg.parquet}")


if __name__ == "__main__":
    cfg, args = from_args(__doc__ or "")
    main(cfg, fixture=args.fixture)

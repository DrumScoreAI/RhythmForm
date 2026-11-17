#!/usr/bin/env python3
"""Download and inspect the RhythmForm synthetic dataset from EIDF S3 storage.

Dataset catalogue:
    https://catalogue.eidf.ac.uk/dataset/eidf262-rhythmform_synthetic_data_v100
S3 endpoint:
    https://s3.eidf.ac.uk/eidf262-rhythmform-synthetic-data-v100

Uses only the Python standard library.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
from typing import Iterable, Iterator
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

DEFAULT_BUCKET_URL = (
    "https://s3.eidf.ac.uk/eidf262-rhythmform-synthetic-data-v100"
)
CHUNK_SIZE = 1024 * 1024  # 1 MiB


@dataclass(frozen=True)
class S3Object:
    key: str
    size: int
    last_modified: str = ""
    etag: str = ""


def _strip_ns(tag: str) -> str:
    """Remove XML namespace prefix from an ElementTree tag."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _format_size(num_bytes: int) -> str:
    """Format byte count as a human-readable string."""
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0 or unit == "TiB":
            return f"{num_bytes} B" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def normalize_bucket_url(bucket_url: str) -> str:
    """Normalize an EIDF S3 bucket URL.

    Publicly published EIDF S3 buckets are addressed without the ``<tenant>:``
    prefix in the path (e.g. ``/eidf262-rhythmform-synthetic-data-v100`` rather
    than ``/eidf262:eidf262-rhythmform-synthetic-data-v100``). If a
    ``tenant:bucket`` path segment is supplied, strip the ``tenant:`` prefix.
    """
    parsed = urllib.parse.urlsplit(bucket_url.rstrip("/"))
    path = parsed.path.lstrip("/")
    if ":" in path:
        first_seg, _, rest = path.partition("/")
        if ":" in first_seg:
            _, _, bucket_name = first_seg.partition(":")
            path = f"{bucket_name}/{rest}" if rest else bucket_name
    return urllib.parse.urlunsplit(
        (parsed.scheme, parsed.netloc, f"/{path}", "", "")
    )


def _build_object_url(bucket_url: str, key: str) -> str:
    """Build a properly URL-encoded object URL for the given S3 key."""
    base = normalize_bucket_url(bucket_url).rstrip("/")
    encoded_key = urllib.parse.quote(key.lstrip("/"), safe="/")
    return f"{base}/{encoded_key}"


def iter_s3_listing(
    bucket_url: str,
    prefix: str = "",
    delimiter: str = "",
    timeout: float = 30.0,
) -> Iterator[tuple[list[S3Object], list[str]]]:
    """Iterate through paginated S3 ListObjects responses.

    Supports both ListObjectsV2 (ContinuationToken) and ListObjectsV1 (Marker).
    Yields tuples of (objects, common_prefixes) per page.
    """
    base = normalize_bucket_url(bucket_url).rstrip("/")
    continuation_token: str | None = None
    marker: str | None = None
    use_v2 = True

    while True:
        params: dict[str, str] = {}
        if use_v2:
            params["list-type"] = "2"
        if prefix:
            params["prefix"] = prefix
        if delimiter:
            params["delimiter"] = delimiter
        if continuation_token:
            params["continuation-token"] = continuation_token
        elif marker:
            params["marker"] = marker

        query = urllib.parse.urlencode(params)
        url = f"{base}?{query}" if query else base

        req = urllib.request.Request(
            url, headers={"User-Agent": "RhythmForm-DataDownloader/1.0"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                xml_bytes = resp.read()
        except urllib.error.HTTPError as exc:
            if use_v2 and exc.code in (400, 501):
                # Fallback to ListObjects v1 if v2 is unsupported
                use_v2 = False
                continuation_token = None
                continue
            raise RuntimeError(
                f"HTTP error {exc.code} while listing {url}: {exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Failed to connect to {url}: {exc.reason}"
            ) from exc

        root = ET.fromstring(xml_bytes)

        objects: list[S3Object] = []
        common_prefixes: list[str] = []
        is_truncated = False
        next_token: str | None = None
        next_marker: str | None = None

        for child in root:
            tag = _strip_ns(child.tag)
            if tag == "Contents":
                key = ""
                size = 0
                last_modified = ""
                etag = ""
                for prop in child:
                    ptag = _strip_ns(prop.tag)
                    text = (prop.text or "").strip()
                    if ptag == "Key":
                        key = text
                    elif ptag == "Size":
                        size = int(text) if text.isdigit() else 0
                    elif ptag == "LastModified":
                        last_modified = text
                    elif ptag == "ETag":
                        etag = text.strip('"')
                if key:
                    objects.append(
                        S3Object(
                            key=key,
                            size=size,
                            last_modified=last_modified,
                            etag=etag,
                        )
                    )
            elif tag == "CommonPrefixes":
                for prop in child:
                    if _strip_ns(prop.tag) == "Prefix" and prop.text:
                        common_prefixes.append(prop.text)
            elif tag == "IsTruncated":
                is_truncated = (child.text or "").strip().lower() == "true"
            elif tag == "NextContinuationToken":
                next_token = (child.text or "").strip()
            elif tag == "NextMarker":
                next_marker = (child.text or "").strip()

        yield objects, common_prefixes

        if not is_truncated:
            break

        if next_token:
            continuation_token = next_token
        elif next_marker:
            marker = next_marker
        elif objects:
            # S3 v1 fallback when NextMarker is omitted
            use_v2 = False
            continuation_token = None
            marker = objects[-1].key
        else:
            break


def print_folder_structure(
    bucket_url: str,
    prefix: str = "",
    show_files: bool = False,
    max_depth: int = 1,
    count_all: bool = False,
    limit: int | None = None,
    timeout: float = 30.0,
) -> None:
    """List the folder hierarchy (and optionally files) of the S3 dataset.

    By default, uses S3 ``delimiter='/'`` so listing the folder structure of
    a 700k-object bucket completes in a single fast request per directory
    rather than enumerating every object in the bucket.
    """
    resolved_url = normalize_bucket_url(bucket_url)
    normalized_prefix = prefix.lstrip("/")

    print(f"Bucket URL : {resolved_url}")
    if normalized_prefix:
        print(f"Prefix     : {normalized_prefix}")
    print("-" * 72)

    if not count_all:
        _print_delimited_tree(
            bucket_url=resolved_url,
            prefix=normalized_prefix,
            current_depth=1,
            max_depth=max_depth,
            show_files=show_files,
            limit=limit,
            timeout=timeout,
        )
        return

    # Full scan mode (--count-all): enumerate every object under prefix
    folder_stats: dict[str, tuple[int, int]] = {}
    all_files: list[S3Object] = []
    base_parts = (
        [p for p in normalized_prefix.strip("/").split("/") if p]
        if normalized_prefix.strip("/")
        else []
    )

    total_files = 0
    total_size = 0
    pages = 0

    for page_objs, _ in iter_s3_listing(
        bucket_url=resolved_url, prefix=normalized_prefix, timeout=timeout
    ):
        pages += 1
        for obj in page_objs:
            if obj.key.endswith("/") and obj.size == 0:
                continue
            total_files += 1
            total_size += obj.size
            if show_files and (limit is None or len(all_files) < limit):
                all_files.append(obj)

            parts = [p for p in obj.key.split("/") if p]
            dir_parts = parts[:-1]
            for depth in range(len(base_parts), len(dir_parts) + 1):
                folder_path = "/".join(dir_parts[:depth])
                if folder_path:
                    folder_path += "/"
                count, size_acc = folder_stats.get(folder_path, (0, 0))
                folder_stats[folder_path] = (count + 1, size_acc + obj.size)

        if pages % 10 == 0:
            print(
                f"  ... scanned {total_files} objects ({_format_size(total_size)})",
                end="\r",
                 file=sys.stderr,
            )

    if pages >= 10:
        print(" " * 72, end="\r", file=sys.stderr)

    print(f"Total files: {total_files} ({_format_size(total_size)})")
    print("-" * 72)

    for folder in sorted(folder_stats.keys()):
        if not folder:
            continue
        rel = folder[len(normalized_prefix) :].strip("/")
        depth = len([p for p in rel.split("/") if p]) if rel else 0
        if max_depth is not None and depth > max_depth:
            continue
        indent = "  " * max(0, depth - 1)
        name = folder.rstrip("/").split("/")[-1] + "/" if depth > 0 else folder
        count, size = folder_stats[folder]
        print(f"{indent}{name:<40} ({count} files, {_format_size(size)})")

    if show_files and all_files:
        print("\nObjects:")
        print("-" * 72)
        for obj in all_files:
            print(f"  {obj.key:<55} {_format_size(obj.size):>12}")
        if limit is not None and total_files > len(all_files):
            print(f"  ... ({total_files - len(all_files)} more objects omitted)")


def _print_delimited_tree(
    bucket_url: str,
    prefix: str,
    current_depth: int,
    max_depth: int,
    show_files: bool,
    limit: int | None,
    timeout: float,
) -> None:
    """Recursively list folders (CommonPrefixes) and files using delimiter='/'."""
    subfolders: list[str] = []
    files: list[S3Object] = []
    total_files_at_level = 0
    total_bytes_at_level = 0

    for page_objs, page_prefixes in iter_s3_listing(
        bucket_url=bucket_url,
        prefix=prefix,
        delimiter="/",
        timeout=timeout,
    ):
        subfolders.extend(page_prefixes)
        for obj in page_objs:
            if obj.key == prefix or (obj.key.endswith("/") and obj.size == 0):
                continue
            total_files_at_level += 1
            total_bytes_at_level += obj.size
            if show_files or current_depth == 1:
                if limit is None or len(files) < limit:
                    files.append(obj)
        if not show_files and limit is not None and total_files_at_level >= limit:
            break

    indent = "  " * (current_depth - 1)

    for folder_prefix in sorted(set(subfolders)):
        folder_name = folder_prefix[len(prefix) :] if prefix else folder_prefix
        print(f"{indent}[DIR]  {folder_name}")
        if current_depth < max_depth:
            _print_delimited_tree(
                bucket_url=bucket_url,
                prefix=folder_prefix,
                current_depth=current_depth + 1,
                max_depth=max_depth,
                show_files=show_files,
                limit=limit,
                timeout=timeout,
            )

    if show_files or (current_depth == 1 and not prefix):
        for obj in files:
            display_name = obj.key[len(prefix) :] if prefix else obj.key
            print(
                f"{indent}[FILE] {display_name:<48} {_format_size(obj.size):>12}"
            )
        if total_files_at_level > len(files):
            omitted = total_files_at_level - len(files)
            print(f"{indent}       ... ({omitted} more files in '{prefix or '/'}')")
    elif total_files_at_level > 0:
        print(
            f"{indent}       ({total_files_at_level} files, "
            f"{_format_size(total_bytes_at_level)} directly in '{prefix}')"
        )


def download_key(
    bucket_url: str,
    key: str,
    output_dir: Path,
    overwrite: bool = False,
    expected_size: int | None = None,
    timeout: float = 60.0,
) -> Path:
    """Download a single S3 object key to `output_dir / key`."""
    clean_key = key.lstrip("/")
    if not clean_key or clean_key.endswith("/"):
        raise ValueError(f"Invalid file key: {key!r}")

    target_path = (output_dir / clean_key).resolve()
    output_root = output_dir.resolve()
    if output_root not in target_path.parents and target_path != output_root:
        raise ValueError(f"Refusing to write outside output directory: {key}")

    if target_path.exists() and not overwrite:
        if expected_size is None or target_path.stat().st_size == expected_size:
            print(f"[SKIP] {clean_key} (already exists)")
            return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".part")

    url = _build_object_url(bucket_url, clean_key)
    req = urllib.request.Request(
        url, headers={"User-Agent": "RhythmForm-DataDownloader/1.0"}
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            with open(tmp_path, "wb") as out_file:
                shutil.copyfileobj(resp, out_file, length=CHUNK_SIZE)
        tmp_path.replace(target_path)
    except urllib.error.HTTPError as exc:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"HTTP {exc.code} downloading '{clean_key}': {exc.reason}"
        ) from exc
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise

    size_str = _format_size(target_path.stat().st_size)
    print(f"[DONE] {clean_key} ({size_str})")
    return target_path


def download_prefix_stream(
    bucket_url: str,
    prefixes: Iterable[str],
    output_dir: Path,
    overwrite: bool = False,
    workers: int = 4,
    timeout: float = 60.0,
) -> tuple[int, int]:
    """Stream S3 listing pages for the given prefixes and download concurrently.

    Downloads begin immediately on the first listing page so large buckets
    (e.g. 700,000+ objects) do not require buffering the entire object list
    in memory before starting transfers.
    """
    resolved_url = normalize_bucket_url(bucket_url)
    succeeded = 0
    failed = 0
    found_any = False

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for raw_prefix in prefixes:
            prefix = raw_prefix.lstrip("/")
            prefix_matched = False

            for page_objs, _ in iter_s3_listing(
                bucket_url=resolved_url, prefix=prefix, timeout=timeout
            ):
                file_objs = [
                    o
                    for o in page_objs
                    if not (o.key.endswith("/") and o.size == 0)
                ]
                if not file_objs:
                    continue

                prefix_matched = True
                found_any = True

                future_to_key = {
                    executor.submit(
                        download_key,
                        bucket_url=resolved_url,
                        key=obj.key,
                        output_dir=output_dir,
                        overwrite=overwrite,
                        expected_size=obj.size,
                        timeout=timeout,
                    ): obj.key
                    for obj in file_objs
                }
                for future in as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        future.result()
                        succeeded += 1
                    except Exception as exc:
                        failed += 1
                        print(f"[FAIL] {key}: {exc}", file=sys.stderr)

            if prefix and not prefix_matched:
                print(
                    f"Warning: no objects found under prefix '{prefix}'.",
                    file=sys.stderr,
                )

    if not found_any:
        print("No matching files to download.")
        return 0, 0

    print(
        f"Finished: {succeeded} succeeded, {failed} failed "
        f"(output: '{output_dir}')."
    )
    return succeeded, failed


def download_folders(
    bucket_url: str,
    folders: Iterable[str],
    output_dir: Path,
    overwrite: bool = False,
    workers: int = 4,
    timeout: float = 60.0,
) -> tuple[int, int]:
    """Download one or more folder prefixes from the S3 bucket."""
    normalized_prefixes = []
    for folder in folders:
        clean = folder.strip("/")
        normalized_prefixes.append(f"{clean}/" if clean else "")
    return download_prefix_stream(
        bucket_url=bucket_url,
        prefixes=normalized_prefixes,
        output_dir=output_dir,
        overwrite=overwrite,
        workers=workers,
        timeout=timeout,
    )


def download_entire_dataset(
    bucket_url: str,
    output_dir: Path,
    overwrite: bool = False,
    workers: int = 4,
    timeout: float = 60.0,
) -> tuple[int, int]:
    """Download all objects in the bucket."""
    return download_prefix_stream(
        bucket_url=bucket_url,
        prefixes=[""],
        output_dir=output_dir,
        overwrite=overwrite,
        workers=workers,
        timeout=timeout,
    )


def download_specific_keys(
    bucket_url: str,
    keys: Iterable[str],
    output_dir: Path,
    overwrite: bool = False,
    workers: int = 4,
    timeout: float = 60.0,
) -> tuple[int, int]:
    """Download specific object keys directly by key name."""
    resolved_url = normalize_bucket_url(bucket_url)
    key_list = [k.lstrip("/") for k in keys if k.strip("/")]
    if not key_list:
        print("No valid keys specified.")
        return 0, 0

    succeeded = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_to_key = {
            executor.submit(
                download_key,
                bucket_url=resolved_url,
                key=key,
                output_dir=output_dir,
                overwrite=overwrite,
                expected_size=None,
                timeout=timeout,
            ): key
            for key in key_list
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                future.result()
                succeeded += 1
            except Exception as exc:
                failed += 1
                print(f"[FAIL] {key}: {exc}", file=sys.stderr)

    print(
        f"Finished: {succeeded} succeeded, {failed} failed "
        f"(output: '{output_dir}')."
    )
    return succeeded, failed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "List or download the RhythmForm Synthetic Dataset (v1.0.0) "
            "from EIDF S3-compatible storage."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # Quickly list top-level folders and root files
  python data_download.py --list

  # List files inside a specific folder (showing first 20 files)
  python data_download.py --list --prefix logs/ --show-files --limit 20

  # Download specific root files or object keys
  python data_download.py --keys all_tokens_corpus.smt full_tokenizer_vocab.json --output ./data

  # Download specific folders (e.g. logs/)
  python data_download.py --folders logs/ --output ./data

  # Download the entire dataset
  python data_download.py --all --output ./data --workers 8
""",
    )

    action_group = parser.add_argument_group("Actions")
    action_group.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List the folder structure and top-level files of the dataset.",
    )
    action_group.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Download the entire dataset.",
    )
    action_group.add_argument(
        "-f",
        "--folders",
        nargs="+",
        metavar="FOLDER",
        help="Download one or more specific folder prefixes.",
    )
    action_group.add_argument(
        "-k",
        "--keys",
        nargs="+",
        metavar="KEY",
        help="Download one or more specific object keys.",
    )

    list_group = parser.add_argument_group("Listing options")
    list_group.add_argument(
        "--prefix",
        default="",
        help="Filter '--list' to a specific key prefix (e.g. 'images/' or 'logs/').",
    )
    list_group.add_argument(
        "--show-files",
        action="store_true",
        help="List individual file keys and sizes under the prefix.",
    )
    list_group.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of files to display per directory with '--list' (default: 50, 0 for unlimited).",
    )
    list_group.add_argument(
        "--max-depth",
        type=int,
        default=1,
        help="Folder recursion depth when using '--list' (default: 1).",
    )
    list_group.add_argument(
        "--count-all",
        action="store_true",
        help="Scan all objects in the bucket/prefix to compute exact per-folder file counts and sizes.",
    )

    dl_group = parser.add_argument_group("Download options")
    dl_group.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("./rhythmform_data"),
        help="Local directory to save downloaded files (default: ./rhythmform_data).",
    )
    dl_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing local files even if sizes match.",
    )
    dl_group.add_argument(
        "-w",
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent download threads (default: 4).",
    )
    dl_group.add_argument(
        "--bucket-url",
        default=DEFAULT_BUCKET_URL,
        help=f"S3 bucket URL (default: {DEFAULT_BUCKET_URL}).",
    )
    dl_group.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP request timeout in seconds (default: 60).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not (args.list or args.all or args.folders or args.keys):
        parser.print_help()
        return 0

    file_limit = None if (args.limit is not None and args.limit <= 0) else args.limit

    try:
        if args.list:
            print_folder_structure(
                bucket_url=args.bucket_url,
                prefix=args.prefix,
                show_files=args.show_files,
                max_depth=args.max_depth,
                count_all=args.count_all,
                limit=file_limit,
                timeout=args.timeout,
            )

        failed = 0
        if args.all:
            _, f = download_entire_dataset(
                bucket_url=args.bucket_url,
                output_dir=args.output,
                overwrite=args.overwrite,
                workers=args.workers,
                timeout=args.timeout,
            )
            failed += f
        else:
            if args.folders:
                _, f = download_folders(
                    bucket_url=args.bucket_url,
                    folders=args.folders,
                    output_dir=args.output,
                    overwrite=args.overwrite,
                    workers=args.workers,
                    timeout=args.timeout,
                )
                failed += f
            if args.keys:
                _, f = download_specific_keys(
                    bucket_url=args.bucket_url,
                    keys=args.keys,
                    output_dir=args.output,
                    overwrite=args.overwrite,
                    workers=args.workers,
                    timeout=args.timeout,
                )
                failed += f

        return 1 if failed > 0 else 0
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())

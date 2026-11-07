from collections import Counter
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

from PIL import Image

from .runtime import write_json


DATASETS = ("vqa_rad", "slake")
ROLES = ("train", "feedback", "test")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decoded_sha256(image):
    image = image.convert("RGB")
    header = f"RGB:{image.width}x{image.height}:".encode()
    return hashlib.sha256(header + image.tobytes()).hexdigest()


def image_path(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
        raise ValueError("Image path must be a safe relative POSIX path")
    absolute = (Path(root) / path).resolve()
    absolute.relative_to(Path(root).resolve())
    return absolute


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL at line {number}: {path}") from error


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def download_assets(sources):
    from huggingface_hub import hf_hub_download

    paths = {}
    for dataset, source in sources.items():
        for split, asset in source["files"].items():
            print(f"Checking {dataset}: {asset['path']}", flush=True)
            local = Path(hf_hub_download(source["repo_id"], asset["path"],
                         repo_type="dataset", revision=source["revision"]))
            if local.stat().st_size != asset["size_bytes"] or file_sha256(local) != asset["sha256"]:
                raise ValueError(f"Source fingerprint mismatch: {dataset}/{split}")
            paths[(dataset, split)] = local
    return paths


def export_source_rows(paths, destination):
    import pyarrow.parquet as parquet

    images = destination / "images"
    images.mkdir()
    records, exclusions = [], []
    cached = {}

    def export_image(content):
        raw_hash = hashlib.sha256(content).hexdigest()
        if raw_hash in cached:
            return cached[raw_hash]
        with Image.open(io.BytesIO(content)) as original:
            if original.width * original.height > 20_000_000:
                raise ValueError("Source image exceeds the declared pixel limit")
            original.load()
            image = original.convert("RGB")
        digest = decoded_sha256(image)
        output = images / (digest + ".png")
        if not output.exists():
            image.save(output, format="PNG")
        value = {"image_path": "images/" + output.name,
                 "image_sha256": digest, "image_file_sha256": file_sha256(output)}
        cached[raw_hash] = value
        return value

    for split in ("train", "test"):
        rows = parquet.read_table(paths[("vqa_rad", split)]).to_pylist()
        for index, row in enumerate(rows):
            answer = str(row["answer"]).strip()
            record = {"id": f"vqa_rad:{split}:{index:06d}", "dataset": "vqa_rad",
                      "source_split": split, "source_index": index,
                      "source_question_id": None, "question": row["question"].strip(),
                      "answer": answer,
                      "answer_type": "closed" if answer.lower() in ("yes", "no") else "open",
                      "answer_type_origin": "reference-derived scoring metadata"}
            record.update(export_image(row["image"]["bytes"]))
            records.append(record)
    with zipfile.ZipFile(paths[("slake", "images")]) as archive:
        members = archive.namelist()
        if len(members) != len(set(members)):
            raise ValueError("SLAKE archive contains duplicate member names")
        names = set(members)
        for split in ("train", "validation", "test"):
            rows = json.loads(paths[("slake", split)].read_text(encoding="utf-8"))
            for index, row in enumerate(rows):
                identity = f"slake:{split}:{row['qid']}"
                if row["q_lang"] != "en":
                    exclusions.append({"id": identity, "dataset": "slake",
                                       "source_split": split, "reason": "non_english"})
                    continue
                if not row["question"].strip() or not str(row["answer"]).strip():
                    if split == "test":
                        raise ValueError(f"Test reference is missing: {identity}; resolve it before evaluation")
                    exclusions.append({"id": identity, "dataset": "slake",
                                       "source_split": split, "reason": "missing_question_or_reference"})
                    continue
                name = row["img_name"]
                matches = [p for p in (name, "imgs/" + name, "./imgs/" + name) if p in names]
                if len(matches) != 1:
                    raise ValueError(f"Missing or ambiguous SLAKE image: {name}")
                record = {"id": identity, "dataset": "slake", "source_split": split,
                          "source_index": index, "source_question_id": row["qid"],
                          "question": row["question"].strip(), "answer": str(row["answer"]).strip(),
                          "answer_type": row["answer_type"].lower(),
                          "answer_type_origin": "source annotation"}
                record.update(export_image(archive.read(matches[0])))
                records.append(record)
    return records, exclusions


def assign_roles(records, seed, rad_feedback_fraction):
    if not 0 < rad_feedback_fraction < 1:
        raise ValueError("rad_feedback_fraction must be between zero and one")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("Source IDs must be unique")
    test_images = {r["image_sha256"] for r in records if r["source_split"] == "test"}
    feedback_images = {r["image_sha256"] for r in records
                       if r["dataset"] == "slake" and r["source_split"] == "validation"}
    feedback_images -= test_images
    rad_images = {r["image_sha256"] for r in records
                  if r["dataset"] == "vqa_rad" and r["source_split"] == "train"} - test_images
    ranked = sorted(rad_images, key=lambda h: hashlib.sha256(f"{seed}:{h}".encode()).hexdigest())
    count = max(1, round(len(ranked) * rad_feedback_fraction)) if ranked else 0
    feedback_images.update(ranked[:count])
    output, exclusions, seen = [], [], {}
    for original in sorted(records, key=lambda r: r["id"]):
        row = dict(original)
        digest = row["image_sha256"]
        if row["source_split"] == "test":
            row["role"] = "test"
        elif digest in test_images:
            exclusions.append({"id": row["id"], "dataset": row["dataset"],
                               "source_split": row["source_split"], "reason": "test_image_overlap"})
            continue
        else:
            row["role"] = "feedback" if digest in feedback_images else "train"
            key = (digest, " ".join(row["question"].lower().split()))
            if key in seen:
                if seen[key] != row["answer"].lower():
                    raise ValueError(f"Conflicting adaptation references: {row['id']}")
                exclusions.append({"id": row["id"], "dataset": row["dataset"],
                                   "source_split": row["source_split"], "reason": "duplicate_adaptation_question"})
                continue
            seen[key] = row["answer"].lower()
        output.append(row)
    return output, exclusions


def validate_records(root, records, verify_images=True):
    ids, hashes, checked = set(), {role: set() for role in ROLES}, {}
    for row in records:
        if row["id"] in ids or row["dataset"] not in DATASETS or row["role"] not in ROLES:
            raise ValueError("Invalid or duplicate manifest identity")
        ids.add(row["id"])
        if not row["question"].strip() or not row["answer"].strip():
            raise ValueError("Question and reference must be nonempty")
        if row["answer_type"] not in ("open", "closed"):
            raise ValueError("Unknown answer type")
        hashes[row["role"]].add(row["image_sha256"])
        path = image_path(root, row["image_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = (row["image_sha256"], row["image_file_sha256"])
        if str(path) in checked and checked[str(path)] != expected:
            raise ValueError("Conflicting image identity in manifests")
        if verify_images and str(path) not in checked:
            if file_sha256(path) != row["image_file_sha256"]:
                raise ValueError(f"Image file changed: {row['id']}")
            with Image.open(path) as image:
                image.load()
                if decoded_sha256(image) != row["image_sha256"]:
                    raise ValueError(f"Decoded image changed: {row['id']}")
        checked[str(path)] = expected
    for left, right in (("train", "feedback"), ("train", "test"), ("feedback", "test")):
        if hashes[left] & hashes[right]:
            raise ValueError(f"Image leakage between {left} and {right}")
    return {role: sum(r["role"] == role for r in records) for role in ROLES}


def prepare_bundle(destination, sources, protocol, paths=None):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Data destination exists. Inspect it or choose a new --data-dir; it will not be overwritten")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".prepare-", dir=destination.parent))
    try:
        downloaded = paths if paths is not None else download_assets(sources)
        rows, excluded = export_source_rows(downloaded, staging)
        records, role_excluded = assign_roles(rows, protocol["seed"], protocol["rad_feedback_fraction"])
        excluded.extend(role_excluded)
        counts = validate_records(staging, records)
        for role in ROLES:
            write_jsonl(staging / (role + ".jsonl"), [r for r in records if r["role"] == role])
        write_jsonl(staging / "exclusions.jsonl", excluded)
        summary = {"bundle_version": 1, "image_hash_protocol": "rgb-dimensions-v1",
                   "sources": sources, "protocol": protocol, "counts": counts,
                   "counts_by_dataset_role": dict(Counter(r["dataset"] + ":" + r["role"] for r in records)),
                   "source_english_rows": len(rows), "exclusion_reasons": dict(Counter(r["reason"] for r in excluded)),
                   "manifest_sha256": {role: file_sha256(staging / (role + ".jsonl")) for role in ROLES},
                   "exclusions_sha256": file_sha256(staging / "exclusions.jsonl"),
                   "limitations": ["Exact decoded-image separation only; no patient-ID or near-duplicate guarantee",
                                   "Adaptation pool differs from the source training partitions",
                                   "VQA-RAD is a pinned processed distribution; SLAKE uses its pinned author distribution"]}
        write_json(staging / "dataset.json", summary)
        staging.rename(destination)
        return summary
    except BaseException:
        shutil.rmtree(staging)
        raise


def load_bundle(root, verify_images=False):
    root = Path(root).resolve()
    summary = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    records = []
    for role in ROLES:
        path = root / (role + ".jsonl")
        if file_sha256(path) != summary["manifest_sha256"][role]:
            raise ValueError(f"Manifest changed: {role}")
        part = list(read_jsonl(path))
        if any(r["role"] != role for r in part):
            raise ValueError("Manifest role and record role disagree")
        records.extend(part)
    counts = validate_records(root, records, verify_images)
    if counts != summary["counts"]:
        raise ValueError("Manifest counts do not match dataset metadata")
    if file_sha256(root / "exclusions.jsonl") != summary["exclusions_sha256"]:
        raise ValueError("Exclusion ledger changed")
    return summary, records

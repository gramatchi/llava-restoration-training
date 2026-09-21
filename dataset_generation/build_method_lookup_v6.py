"""Merges the best-method lookups of all 3000 photos.

The original 1000 photos come from original_1000_method_lookup_*.json (keys are
bare filenames, since they sit directly in data/raw). The other 2000 come from
new_2000_per_image_*.json (output of find_best_methods_*_v6.py); their keys
become paths relative to data/raw, e.g. extra_images/0002000/0001001.png, so
the manifest builder can use them as-is.

Output: method_lookup_{jpeg,noise}.json as {image_path: {severity_value: method}},
where the method is the one with the best SSIM.
"""
import json
import pathlib

DATA = pathlib.Path(__file__).resolve().parent / "data"

# photo id range -> subfolder under data/raw
BATCH_DIRS = {
    range(1001, 2001): "extra_images/0002000",
    range(2001, 3001): "extra_images/0003000",
}


def _relative_path_for(bare_name):
    num = int(bare_name.split(".")[0])
    if num <= 1000:
        return bare_name
    for id_range, subdir in BATCH_DIRS.items():
        if num in id_range:
            return f"{subdir}/{bare_name}"
    raise ValueError(f"photo id {num} ({bare_name}) is outside 1-3000")


def _reduce_new_results(per_image_list, value_field):
    out = {}
    for entry in per_image_list:
        path = _relative_path_for(entry["image"])
        by_value = entry[f"by_{value_field}"]
        out[path] = {v: data["best_ssim"]["method"] for v, data in by_value.items()}
    return out


def merge(dtype, value_field):
    old = json.loads((DATA / f"original_1000_method_lookup_{dtype}.json").read_text())
    new_raw = json.loads((DATA / f"new_2000_per_image_{dtype}.json").read_text())
    new = _reduce_new_results(new_raw, value_field)

    overlap = set(old) & set(new)
    if overlap:
        raise ValueError(f"old and new {dtype} lookups share keys: {overlap}")

    merged = {**old, **new}
    out_path = DATA / f"method_lookup_{dtype}.json"
    out_path.write_text(json.dumps(merged, indent=2))
    print(f"{dtype}: old={len(old)} new={len(new)} merged={len(merged)} -> {out_path}")
    return merged


if __name__ == "__main__":
    jpeg = merge("jpeg", "quality")
    noise = merge("noise", "sigma")
    assert set(jpeg) == set(noise), "jpeg and noise lookups cover different photos"
    print(f"\nPhotos covered: {len(jpeg)}")

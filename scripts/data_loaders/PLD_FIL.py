# Builds a Hugging Face dataset from the UP-DSP-PLD Filipino (FIL) recordings and the IPA labels
# produced by notebooks/EDA/PLD_FIL_EDA_G2P.ipynb (.data/MDC/UP-DSP-PLD/PLD/FIL_g2p.csv).
# Usage: python ./scripts/data_loaders/PLD_FIL.py <hf_repo_id | local_output_dir> [--report] [--public]
#   hf_repo_id (e.g. your-username/PLD-FIL) pushes to the hub (private unless --public); log in first with `hf auth login`
#   a path starting with "." or "/" saves to disk instead (used for local testing)
#   --report only prints the split and its balance checks
#
# Splits (see split_speakers and held_out_texts):
#   - speaker-disjoint, 80/10/10 of each gender x age-band group's hours for train/validation/test; of N_SPLIT_DRAWS
#     seeded random draws, the one whose splits best match the whole corpus (gender, age band, province, prompt
#     category, label source; summed total variation distance) is kept
#   - 10% of the distinct prompts of every category are never used for training, so validation and test contain
#     prompts the model has not heard (text_in_train = False) as well as prompts it has

import os
import io
import sys

import numpy as np
import pandas as pd
from datasets import Audio, Dataset, DatasetDict, Features, Value

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.audio import TARGET_SAMPLE_RATE

PLD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", ".data", "MDC", "UP-DSP-PLD", "PLD")
FIL_DIR = os.path.join(PLD_DIR, "FIL")
G2P_CSV = os.path.join(PLD_DIR, "FIL_g2p.csv")

SEED = 42
SPLIT_FRACTIONS = {"test": 0.10, "validation": 0.10, "train": 0.80}  # of each gender x age-band group's hours
HELD_OUT_TEXT_FRACTION = 0.10  # of each category's distinct prompts, never used for training
N_SPLIT_DRAWS = 20000
AGE_BANDS = [0, 18, 20, 200]  # 16-18, 19-20, 21+: roughly equal thirds of the speakers

FEATURES = Features(
    {
        "audio": {"bytes": Value("binary"), "path": Value("string")},  # FLAC, cast to Audio after building
        "ipa": Value("string"),
        "text": Value("string"),
        "spoken_text": Value("string"),
        "raw_text": Value("string"),
        "id": Value("string"),
        "category": Value("string"),
        "prompt_file": Value("string"),
        "word_sources": Value("string"),
        "has_manual_reading": Value("bool"),
        "duration": Value("float32"),
        "speaker_id": Value("string"),
        "speaker_gender": Value("string"),
        "speaker_age": Value("int32"),
        "speaker_dialect": Value("string"),
        "label_source": Value("string"),
        "text_in_train": Value("bool"),
    }
)


def load_usable_rows():
    """Rows recommended for training in the G2P notebook (section 8): readable, non-empty audio with a reviewed IPA label."""
    df = pd.read_csv(G2P_CSV)
    usable = (
        (df.wav_readable == True)
        & (df.empty_audio == False)
        & (df.needs_review == False)
        & df.ipa.notna()
    )
    df = df[usable].reset_index(drop=True)
    text_cols = df.columns[df.dtypes == object]
    df[text_cols] = df[text_cols].fillna("")
    return df


def label_source(word_sources):
    return word_sources.str.contains("epitran|hyphen").map({True: "rule-based", False: "confirmed"})


def province(dialect):
    # "Tagalog - Laguna", "Tagalog-Laguna", "Tagalog Laguna" -> "Laguna"
    return dialect.str.replace(r"^Tagalog\s*-?\s*", "", regex=True).str.strip().replace("", "unknown")


def speaker_table(df):
    spk = df.groupby("SpeakerID").agg(
        gender=("SpeakerGender", "first"), age=("SpeakerAge", "first"), dialect=("SpeakerDialect", "first"), hours=("duration_s", lambda d: d.sum() / 3600)
    )
    spk["age_band"] = pd.cut(spk.age, AGE_BANDS, labels=["16-18", "19-20", "21+"])
    top = province(spk.dialect).value_counts()
    spk["province"] = province(spk.dialect).where(lambda p: p.map(top) >= 5, "other")
    return spk


def draw_split(groups, rng, n_speakers):
    """One random speaker split, as split codes (index into SPLIT_FRACTIONS) per speaker position. Within each
    gender x age-band group, speakers in random order fill test, then validation, up to their share of the group's
    hours; the rest is train."""
    codes = np.full(n_speakers, list(SPLIT_FRACTIONS).index("train"))
    for positions, hours in groups:
        order = rng.permutation(len(positions))
        filled = {"test": 0.0, "validation": 0.0}
        for i in order:
            for k in ("test", "validation"):
                if filled[k] + hours[i] / 2 <= SPLIT_FRACTIONS[k] * hours.sum():
                    codes[positions[i]], filled[k] = list(SPLIT_FRACTIONS).index(k), filled[k] + hours[i]
                    break
    return codes


BALANCE_ATTRIBUTES = ["gender", "age_band", "province", "category", "label_source"]


def balance_features(df, spk):
    """Per speaker, hours by value of every balance attribute (one block of columns per attribute)."""
    rows = df.join(spk[["gender", "age_band", "province"]], on="SpeakerID")
    blocks = [rows.pivot_table(index="SpeakerID", columns=col, values="duration_s", aggfunc="sum", fill_value=0, observed=True) for col in BALANCE_ATTRIBUTES]
    return np.hstack([b.reindex(spk.index).to_numpy() for b in blocks]), np.cumsum([0] + [b.shape[1] for b in blocks])


def imbalance(codes, features, bounds):
    """Summed total variation distance between each split's attribute distributions and the whole corpus's, plus the
    deviation of each split's share of hours from its target."""
    overall = features.sum(0)
    score = 0.0
    for k, (name, fraction) in enumerate(SPLIT_FRACTIONS.items()):
        part = features[codes == k].sum(0)
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            score += 0.5 * np.abs(part[lo:hi] / max(part[lo:hi].sum(), 1e-9) - overall[lo:hi] / overall[lo:hi].sum()).sum()
        score += abs(part[: bounds[1]].sum() / overall[: bounds[1]].sum() - fraction)
    return score


def split_speakers(df):
    """Speaker-disjoint train/validation/test split: the most balanced of N_SPLIT_DRAWS seeded random draws."""
    spk = speaker_table(df)
    features, bounds = balance_features(df, spk)
    groups = [(np.flatnonzero(spk.index.isin(g.index)), g.hours.to_numpy()) for _, g in spk.groupby(["gender", "age_band"], observed=True)]
    rng = np.random.default_rng(SEED)
    best, best_score = None, np.inf
    for _ in range(N_SPLIT_DRAWS):
        codes = draw_split(groups, rng, len(spk))
        score = imbalance(codes, features, bounds)
        if score < best_score:
            best, best_score = codes, score
    split_of = dict(zip(spk.index, np.array(list(SPLIT_FRACTIONS))[best]))
    return df.SpeakerID.map(split_of)


def held_out_texts(df):
    """HELD_OUT_TEXT_FRACTION of each category's distinct prompts (seeded), never used for training."""
    rng = np.random.default_rng(SEED)
    held = set()
    for _, texts in df.groupby("category").text.unique().items():
        texts = sorted(texts)
        held |= set(rng.choice(texts, size=round(len(texts) * HELD_OUT_TEXT_FRACTION), replace=False))
    return held


def assign_splits(df):
    df = df.copy()
    df["label_source"] = label_source(df.word_sources)
    df["split"] = split_speakers(df)
    held = held_out_texts(df)
    df.loc[(df.split == "train") & df.text.isin(held), "split"] = "unused"  # held-out prompts never reach training
    df["text_in_train"] = df.text.isin(set(df.loc[df.split == "train", "text"]))
    return df


def report(df):
    from scipy.stats import chi2_contingency, mannwhitneyu

    used = df[df.split != "unused"]
    spk = speaker_table(used).assign(split=used.groupby("SpeakerID").split.first())
    table = used.groupby("split").agg(
        utterances=("wav", "size"), hours=("duration_s", lambda d: round(d.sum() / 3600, 2)), speakers=("SpeakerID", "nunique"),
        unseen_prompt_share=("text_in_train", lambda t: round(1 - t.mean(), 3)),
    ).loc[["train", "validation", "test"]]
    table["hours share"] = (table.hours / table.hours.sum()).round(3)
    table["women"] = spk[spk.gender == "female"].groupby("split").size()
    table["median age"] = spk.groupby("split").age.median()
    print(table.to_string())
    print("\nbalance checks (chi-square / Mann-Whitney p-values; a large p means no evidence that the splits differ):")
    for col in ["gender", "age_band", "province"]:
        print(f"  speakers' {col:9}: p = {chi2_contingency(pd.crosstab(spk[col], spk.split))[1]:.2f}")
    for col in ["category", "label_source"]:
        print(f"  utterances' {col:12}: p = {chi2_contingency(pd.crosstab(used[col], used.split))[1]:.2f}")
    for split in ["validation", "test"]:
        p = mannwhitneyu(used[used.split == split].duration_s, used[used.split == "train"].duration_s).pvalue
        print(f"  durations {split} vs train: p = {p:.2f}")
    print(f"\nheld-out prompts: {(df.split == 'unused').sum()} training utterances ({df.loc[df.split == 'unused', 'duration_s'].sum() / 3600:.1f} h) not used")


def flac_bytes(wav_path):
    import soundfile as sf  # imported here: datasets cannot hash a generator that references the module

    audio, sr = sf.read(wav_path, dtype="int16")
    assert sr == TARGET_SAMPLE_RATE and audio.ndim == 1, wav_path
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format="FLAC", subtype="PCM_16")
    return buffer.getvalue()


def gen_rows(rows):
    for r in rows.itertuples():
        yield {
            "audio": {"bytes": flac_bytes(os.path.join(FIL_DIR, r.wav)), "path": r.wav.replace(".wav", ".flac")},
            "ipa": r.ipa,
            "text": r.text,
            "spoken_text": r.spoken_text,
            "raw_text": r.raw_text,
            "id": os.path.splitext(os.path.basename(r.wav))[0],
            "category": r.category,
            "prompt_file": r.prompt_file,
            "word_sources": r.word_sources,
            "has_manual_reading": bool(r.has_manual_reading),
            "duration": float(r.duration_s),
            "speaker_id": f"{int(r.SpeakerID):04d}",
            "speaker_gender": r.SpeakerGender,
            "speaker_age": int(r.SpeakerAge) if str(r.SpeakerAge).isdigit() else -1,
            "speaker_dialect": r.SpeakerDialect,
            "label_source": r.label_source,
            "text_in_train": bool(r.text_in_train),
        }


def build_dataset_dict(df):
    return DatasetDict(
        {
            split: Dataset.from_generator(gen_rows, features=FEATURES, gen_kwargs={"rows": df[df.split == split]}).cast_column("audio", Audio(sampling_rate=TARGET_SAMPLE_RATE))  # type: ignore
            for split in ["train", "validation", "test"]
        }
    )


def main(args):
    if len(args) < 1:
        print("Usage: python ./scripts/data_loaders/PLD_FIL.py <hf_repo_id | local_output_dir> [--report] [--public]")
        return
    target = args[0]
    df = assign_splits(load_usable_rows())
    report(df)
    if "--report" in args:
        return
    ds = build_dataset_dict(df)
    if target.startswith((".", "/")):
        ds.save_to_disk(target)
        print("saved to", target)
    else:
        ds.push_to_hub(target, private="--public" not in args, max_shard_size="500MB")
        print("pushed to https://huggingface.co/datasets/" + target)


if __name__ == "__main__":
    main(sys.argv[1:])
